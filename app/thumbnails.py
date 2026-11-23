"""Optimal thumbnail-plan solver for one video's playable shots.

Problem
-------
Given the playable shot numbers in playback order and a thumbnail count K
(1..8), split the playable shots into exactly K non-empty contiguous groups.
A group may never span a restricted shot, so it stays inside one maximal run
of consecutive playable shot numbers.  One shot per group becomes its
thumbnail.  Up to two requested shot numbers must be chosen as thumbnails.

Cost of a plan is the sum, over every playable shot, of the squared Euclidean
distance between that shot's feature and its group's thumbnail feature.  The
optimum is the minimum total cost; ties are broken by the lexicographically
smallest list of thumbnail shot numbers in playback order.

Algorithm (no partition enumeration)
------------------------------------
Let positions 0..M-1 be the playable shots and cost(p, j, q) the cost of
group [p, j) represented by position q.  Standard contiguous-partition DP:

    dp[g][j] = min over p < j, q in [p, j)
               dp[g-1][p] + cost(p, j, q)

Only groups without an internal gap are admissible (the shot numbers of
positions p..j-1 must be consecutive, so no group ever spans a restricted
shot), and a group containing a required shot
must use that shot as its representative (two required shots in one group is
infeasible).  For every (p, j) the best q is therefore either forced or found
by a linear scan, giving an O(M^3) segment table for M <= 200 (~1.3M O(1)
lookups) and an O(K*M^2) DP.  Nothing enumerates the C(M-1, K-1) partitions.

Exact ties
----------
Every IEEE-754 double is a dyadic rational.  All feature coordinates are
scaled by one common power of two into exact integers, so squared distances
and their prefix sums are exact Python integers and equal costs compare equal
bit-for-bit.  No epsilon is used, and the lexicographic tie-break is exact.
"""

import time
from bisect import bisect_left

THUMB_MIN = 1
THUMB_MAX = 8
MAX_PLAN_SHOTS = 200


class ThumbnailPlan:
    """Result of a successful solve."""

    __slots__ = ("segments", "thumbnail_count", "required_shots",
                 "total_cost_scaled", "scale_exponent")

    def __init__(self, segments, thumbnail_count, required_shots,
                 total_cost_scaled, scale_exponent):
        self.segments = segments            # [{"start","end","shot"}, ...]
        self.thumbnail_count = thumbnail_count
        self.required_shots = required_shots
        self.total_cost_scaled = total_cost_scaled  # exact integer
        self.scale_exponent = scale_exponent        # cost = int / 2**(2*exp)

    def thumbnail_shots(self):
        return [seg["shot"] for seg in self.segments]


def _scaled_features(features, visible):
    """Exact integer features for the playable shots.

    Returns (scaled, exponent) where scaled[i] is a list of ints and the real
    value of coordinate x is scaled / 2**exponent.
    """
    ratios = []
    exp = 0
    for shot in visible:
        row = []
        for x in features[shot]:
            num, den = float(x).as_integer_ratio()
            shift = den.bit_length() - 1
            if shift > exp:
                exp = shift
            row.append((num, shift))
        ratios.append(row)
    return [[num << (exp - shift) for num, shift in row] for row in ratios], exp


def _gap_free(visible, p, j):
    """True if playable positions [p, j) are consecutive shot numbers."""
    return visible[j - 1] - visible[p] == (j - 1) - p


def optimal_plan(features, visible, k, required=()):
    """Compute the minimum-cost K-thumbnail plan.

    features : sequence indexed by shot number, each a sequence of floats
    visible  : ascending playable shot numbers
    k        : number of thumbnails / groups, 1..8
    required : shot numbers that must become thumbnails

    Returns ThumbnailPlan or None when no feasible plan exists (K larger than
    the playable count, K smaller than the number of playable runs, a
    required shot not playable, two required shots forced into one group, ...).
    """
    visible = list(visible)
    m = len(visible)
    req = sorted(set(required))
    if k < THUMB_MIN or k > THUMB_MAX or k > m:
        return None
    if len(req) > k:
        return None
    pos_of = {shot: i for i, shot in enumerate(visible)}
    req_pos = [pos_of[s] for s in req if s in pos_of]
    if len(req_pos) != len(req):
        return None  # a required shot is restricted / out of range

    # Every maximal run of consecutive playable shots needs its own group(s).
    runs = 1
    for i in range(1, m):
        if visible[i] != visible[i - 1] + 1:
            runs += 1
    if k < runs:
        return None

    scaled, exp = _scaled_features(features, visible)

    # pref[q][j] = sum_{i < j} ||scaled[i] - scaled[q]||^2  (exact integers)
    # Built row by row; yield the GIL periodically so the service can keep
    # accepting imports/orders while a plan is being computed.
    pref = []
    for q in range(m):
        fq = scaled[q]
        row = [0] * (m + 1)
        acc = 0
        for i in range(m):
            fi = scaled[i]
            acc += sum((a - b) * (a - b) for a, b in zip(fi, fq))
            row[i + 1] = acc
        pref.append(row)
        time.sleep(0)

    # seg[p][j] = (cost, q) for the best group over positions [p, j), or None.
    seg = [[None] * (m + 1) for _ in range(m)]
    for p in range(m):
        segp = seg[p]
        for j in range(p + 1, m + 1):
            if not _gap_free(visible, p, j):
                continue
            lo = bisect_left(req_pos, p)
            hi = bisect_left(req_pos, j)
            nreq = hi - lo
            if nreq >= 2:
                continue
            if nreq == 1:
                q = req_pos[lo]
                segp[j] = (pref[q][j] - pref[q][p], q)
                continue
            best_cost = None
            best_q = -1
            for q in range(p, j):
                row = pref[q]
                cost = row[j] - row[p]
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_q = q
            segp[j] = (best_cost, best_q)
        time.sleep(0)  # keep the service responsive during the O(N^3) scan

    # dp[g][j] = (cost, reps tuple, boundaries tuple) covering positions [0,j)
    # with g groups.  Ranking is (cost, reps): exact-integer cost, then
    # lexicographically smallest representative list.  For equal (p, j) the
    # segment scan above already keeps the smallest q among equal costs.
    dp = [[None] * (m + 1) for _ in range(k + 1)]
    dp[0][0] = (0, (), ())
    for g in range(1, k + 1):
        dpg = dp[g]
        prev = dp[g - 1]
        for j in range(g, m + 1):
            best = None
            for p in range(g - 1, j):
                base = prev[p]
                if base is None:
                    continue
                s = seg[p][j]
                if s is None:
                    continue
                cost = base[0] + s[0]
                reps = base[1] + (s[1],)
                if best is None or cost < best[0] or (cost == best[0] and reps < best[1]):
                    best = (cost, reps, base[2] + (j,))
            dpg[j] = best
        time.sleep(0)

    final = dp[k][m]
    if final is None:
        return None
    cost, reps, bounds = final

    segments = []
    start_pos = 0
    for g in range(k):
        end_pos = bounds[g]
        q = reps[g]
        segments.append({
            "start": visible[start_pos],          # inclusive, 0-based
            "end": visible[end_pos - 1] + 1,      # exclusive, 0-based
            "shot": visible[q],
        })
        start_pos = end_pos

    return ThumbnailPlan(segments, k, req, cost, exp)
