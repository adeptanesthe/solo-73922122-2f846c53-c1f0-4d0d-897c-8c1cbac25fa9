"""Self-tests: hiding/recovery correctness, overlap semantics, idempotency,
conflict handling, withdrawn marker, <=1e-9 agreement with a stable full
recomputation, and O(log N) update timing."""

import math
import random
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.segment_tree import DIM, VisibleSoftmaxTree


# ------------------------------------------------------------- reference

def brute_force_vector(features, scores, hidden):
    """Stable full recomputation: max-shift + math.fsum (exact-ish sums)."""
    vis = [i for i in range(len(scores)) if i not in hidden]
    if not vis:
        return None
    m = max(scores[i] for i in vis)
    ws = [(i, math.exp(scores[i] - m)) for i in vis]
    s = math.fsum(w for _, w in ws)
    return [math.fsum(w * features[i][k] for i, w in ws) / s for k in range(DIM)]


def assert_close(tree_vec, ref_vec, tol=1e-9):
    if ref_vec is None:
        assert tree_vec is None
        return
    assert tree_vec is not None
    diffs = [abs(a - b) for a, b in zip(tree_vec, ref_vec)]
    assert max(diffs) <= tol, f"max dim diff {max(diffs):.3e} > {tol}"


# ---------------------------------------------------- tree vs brute force

@pytest.mark.parametrize("seed", range(40))
def test_tree_matches_stable_full_recompute(seed):
    rng = random.Random(seed)
    n = rng.choice([1, 2, 3, 5, 17, 64, 100, 333, 1000])
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))

    active = []  # (start, end) of live orders
    for _ in range(30):
        if active and rng.random() < 0.4:
            i = rng.randrange(len(active))
            s, e = active.pop(i)
            tree.add_cover(s, e, -1)
            for j in range(s, e):
                hidden.discard(j)  # rebuilt below anyway
        else:
            s = rng.randrange(0, n)
            e = rng.randrange(s + 1, n + 1)
            tree.add_cover(s, e, +1)
            active.append((s, e))
        # rebuild the hidden set from scratch (reference counting semantics)
        hidden = set()
        for s, e in active:
            hidden.update(range(s, e))
        assert tree.visible_count() == n - len(hidden)
        assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))


def test_tree_extreme_scores_and_features():
    rng = random.Random(7)
    n = 500
    features = [[rng.choice([-1.0, 1.0]) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.choice([-1000.0, 1000.0]) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    active = []
    for _ in range(20):
        s = rng.randrange(0, n)
        e = rng.randrange(s + 1, n + 1)
        tree.add_cover(s, e, +1)
        active.append((s, e))
        hidden.update(range(s, e))
        assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))
    # revoke everything -> must recover the full-set vector exactly-ish
    for s, e in active:
        tree.add_cover(s, e, -1)
    assert tree.visible_count() == n
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, set()))


def test_tree_all_hidden_returns_none():
    tree = VisibleSoftmaxTree([[0.5] * DIM, [-0.5] * DIM], [3.0, -3.0])
    tree.add_cover(0, 2, +1)
    assert tree.visible_count() == 0
    assert tree.retrieval_vector() is None
    tree.add_cover(0, 2, -1)
    assert tree.visible_count() == 2
    assert tree.retrieval_vector() is not None


def test_tree_large_n_accuracy():
    rng = random.Random(99)
    n = 100_000
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    for _ in range(300):
        s = rng.randrange(0, n)
        e = min(n, s + rng.randrange(1, 5000))
        tree.add_cover(s, e, +1)
        hidden.update(range(s, e))
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))


# ------------------------------------------------------------------- API

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _shots(n, seed=0):
    rng = random.Random(seed)
    return [
        {"feature": [rng.uniform(-1, 1) for _ in range(DIM)],
         "score": rng.uniform(-1000, 1000)}
        for _ in range(n)
    ]


def test_api_hide_overlap_revoke_recover(client):
    shots = _shots(8, seed=1)
    r = client.post("/videos/v1/shots", json={"shots": shots})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["visible_count"] == 8
    initial = body["vector"]

    feats = [s["feature"] for s in shots]
    scores = [s["score"] for s in shots]

    # order A hides [2,6), order B hides [4,8)  (overlap [4,6))
    assert client.post("/videos/v1/restrictions",
                       json={"order_id": "A", "start": 2, "end": 6}).json()["visible_count"] == 4
    assert client.post("/videos/v1/restrictions",
                       json={"order_id": "B", "start": 4, "end": 8}).json()["visible_count"] == 2

    # duplicate registration of A: no state change
    dup = client.post("/videos/v1/restrictions", json={"order_id": "A", "start": 2, "end": 6})
    assert dup.status_code == 200 and dup.json()["visible_count"] == 2

    # same id, different range: conflict
    conflict = client.post("/videos/v1/restrictions", json={"order_id": "A", "start": 2, "end": 7})
    assert conflict.status_code == 409

    # revoke A: [2,4) playable again, [4,8) still hidden by B
    r = client.delete("/videos/v1/restrictions/A")
    body = r.json()
    assert body["visible_count"] == 4
    ref = brute_force_vector(feats, scores, set(range(4, 8)))
    assert max(abs(a - b) for a, b in zip(body["vector"], ref)) <= 1e-9

    # revoke B: fully recovered, vector identical to the import-time one
    r = client.delete("/videos/v1/restrictions/B")
    body = r.json()
    assert body["visible_count"] == 8
    assert max(abs(a - b) for a, b in zip(body["vector"], initial)) <= 1e-9

    # unknown order / unknown video
    assert client.delete("/videos/v1/restrictions/B").status_code == 404
    assert client.get("/videos/nope/retrieval-vector").status_code == 404


def test_api_withdrawn_marker(client):
    client.post("/videos/v2/shots", json={"shots": _shots(3, seed=2)})
    r = client.post("/videos/v2/restrictions", json={"order_id": "ALL", "start": 0, "end": 3})
    body = r.json()
    assert body["status"] == "withdrawn" and body["vector"] is None and body["visible_count"] == 0
    r = client.delete("/videos/v2/restrictions/ALL")
    body = r.json()
    assert body["status"] == "ok" and body["vector"] is not None


def test_api_validation(client):
    client.post("/videos/v3/shots", json={"shots": _shots(4, seed=3)})
    bad_dim = {"feature": [0.0] * 15, "score": 0.0}
    assert client.post("/videos/v4/shots", json={"shots": [bad_dim]}).status_code == 422
    bad_val = {"feature": [1.5] * DIM, "score": 0.0}
    assert client.post("/videos/v4/shots", json={"shots": [bad_val]}).status_code == 422
    bad_score = {"feature": [0.0] * DIM, "score": 1000.1}
    assert client.post("/videos/v4/shots", json={"shots": [bad_score]}).status_code == 422
    assert client.post("/videos/v3/restrictions",
                       json={"order_id": "X", "start": 3, "end": 3}).status_code == 400
    assert client.post("/videos/v3/restrictions",
                       json={"order_id": "X", "start": 0, "end": 5}).status_code == 400
    assert client.post("/videos/v3/shots", json={"shots": []}).status_code == 422


# ------------------------------------------------------------- performance

def test_update_is_log_n():
    """Register/revoke must stay O(log N): per-op cost at N=100k should be
    roughly the same as at N=10k (factor << the 10x linear growth)."""
    timings = {}
    for n in (10_000, 100_000):
        rng = random.Random(5)
        features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
        scores = [rng.uniform(-1000, 1000) for _ in range(n)]
        tree = VisibleSoftmaxTree(features, scores)
        ops = 300
        ranges = []
        for _ in range(ops):
            s = rng.randrange(0, n)
            e = min(n, s + rng.randrange(1, 2000))
            ranges.append((s, e))
        t0 = time.perf_counter()
        for s, e in ranges:
            tree.add_cover(s, e, +1)
        for s, e in ranges:
            tree.add_cover(s, e, -1)
        dt = time.perf_counter() - t0
        timings[n] = dt / (2 * ops)
        print(f"\nN={n}: {ops} register + {ops} revoke in {dt:.3f}s "
              f"-> {dt / (2 * ops) * 1e6:.1f} us/op")
    assert timings[100_000] < 5e-3  # generous absolute bound: 5 ms per op
    assert timings[100_000] < 6 * timings[10_000]  # far below 10x linear growth


# --------------------------------------------------------- thumbnail plans

from app.thumbnails import optimal_plan

def _wait_plan(client, vid, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/videos/{vid}/thumbnail-plan").json()
        if not body["pending"]:
            return body
        time.sleep(0.005)
    raise AssertionError("plan computation did not finish")


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _plan_cost(feats, segments):
    total = 0.0
    for seg in segments:
        rep = feats[seg["shot"]]
        for i in range(seg["start"], seg["end"]):
            total += _dist(feats[i], rep)
    return total


def _brute_plan(feats, visible, k, required):
    """Reference: enumerate every split of the playable sequence."""
    import itertools
    m = len(visible)
    req = set(required)
    if k < 1 or k > m or len(req) > k or any(s not in visible for s in req):
        return None
    best = None
    for cuts in itertools.combinations(range(1, m), k - 1):
        bounds = (0,) + cuts + (m,)
        reps, total, ok = [], 0.0, True
        for g in range(k):
            p, j = bounds[g], bounds[g + 1]
            if visible[j - 1] - visible[p] != (j - 1) - p:
                ok = False  # group would cross a restricted shot
                break
            poss = [i for i in range(p, j) if visible[i] in req]
            if len(poss) > 1:
                ok = False
                break
            if poss:
                q = poss[0]
            else:
                q = min(range(p, j), key=lambda q: (
                    sum(_dist(feats[visible[q]], feats[visible[i]]) for i in range(p, j)), q))
            reps.append(visible[q])
            total += sum(_dist(feats[visible[q]], feats[visible[i]]) for i in range(p, j))
        if not ok or not all(r in reps for r in req):
            continue
        if best is None or (round(total, 9), reps) < (round(best[0], 9), best[1]):
            best = (total, reps)
    return None if best is None else best[1]


@pytest.mark.parametrize("seed", range(25))
def test_solver_matches_bruteforce(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 9)
    feats = [[rng.choice([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]) for _ in range(3)]
             for _ in range(n)]
    visible = [i for i in range(n) if rng.random() < 0.8]
    if not visible:
        return
    k = rng.randint(1, min(4, len(visible)))
    required = rng.sample(visible, rng.randint(0, min(2, len(visible))))
    plan = optimal_plan(feats, visible, k, required)
    expected = _brute_plan(feats, visible, k, required)
    if expected is None:
        assert plan is None
        return
    assert plan is not None
    assert plan.thumbnail_shots() == expected
    got = plan.total_cost_scaled / (2 ** (2 * plan.scale_exponent))
    # verify the reported cost against a direct recomputation
    direct = _plan_cost(feats, plan.segments)
    assert abs(got - direct) <= 1e-9


def test_solver_respects_restricted_boundaries():
    # shots 3,4 restricted: playable runs are [0,3) and [5,9)
    feats = [[float(i)] + [0.0] * 15 for i in range(10)]
    visible = [0, 1, 2, 5, 6, 7, 8, 9]
    # K=1 is infeasible: a group would cross the restricted shots
    assert optimal_plan(feats, visible, 1) is None
    plan = optimal_plan(feats, visible, 2)
    assert plan is not None
    for seg in plan.segments:
        assert not (seg["start"] <= 2 and seg["end"] > 3)  # no crossing
        assert visible[0] <= seg["shot"] < seg["end"]


def test_solver_requires_requested_shots():
    feats = [[float(i)] + [0.0] * 15 for i in range(6)]
    visible = list(range(6))
    plan = optimal_plan(feats, visible, 3, required=[1, 4])
    assert plan is not None
    assert {1, 4} <= set(plan.thumbnail_shots())
    # a required restricted shot is infeasible
    assert optimal_plan(feats, [0, 1, 3, 4, 5], 2, required=[2]) is None
    # more required shots than thumbnails
    assert optimal_plan(feats, visible, 1, required=[1, 4]) is None


def test_solver_lexicographic_tiebreak():
    # all features identical -> every plan ties; expect the earliest reps
    feats = [[0.0] * 16 for _ in range(6)]
    plan = optimal_plan(feats, list(range(6)), 3)
    assert plan.thumbnail_shots() == [0, 1, 2]
    assert [s["end"] for s in plan.segments] == [1, 2, 6]


def test_plan_api_is_async_and_importable(client):
    shots = _shots(40, seed=11)
    assert client.post("/videos/p1/shots", json={"shots": shots}).status_code == 200
    r = client.post("/videos/p1/thumbnail-plan",
                    json={"thumbnail_count": 3, "required_shots": [0, 39]})
    assert r.status_code == 200
    assert r.json()["status"] == "computing"
    # import and restriction endpoints stay usable while it computes
    assert client.get("/videos/p1/retrieval-vector").status_code == 200
    body = _wait_plan(client, "p1")
    assert body["status"] == "published"
    assert body["shots"][0] == 0 and body["shots"][-1] == 39
    assert len(body["segments"]) == 3
    # segments are half-open, contiguous over the playable shots
    assert body["segments"][0]["start"] == 0
    assert body["segments"][-1]["end"] == 40
    for a, b in zip(body["segments"], body["segments"][1:]):
        assert a["end"] == b["start"]
    for seg in body["segments"]:
        assert seg["start"] <= seg["shot"] < seg["end"]
    feats = [s["feature"] for s in shots]
    assert abs(_plan_cost(feats, body["segments"])
               - _plan_cost(feats, body["segments"])) == 0


def test_plan_api_withdrawal_on_visibility_change_only(client):
    shots = _shots(12, seed=12)
    client.post("/videos/p2/shots", json={"shots": shots})
    client.post("/videos/p2/thumbnail-plan", json={"thumbnail_count": 2})
    body = _wait_plan(client, "p2")
    assert body["status"] == "published"
    gen = body["generation"]

    # overlapping order hides [2,6) -> playable set changes -> plan withdrawn
    client.post("/videos/p2/restrictions", json={"order_id": "A", "start": 2, "end": 6})
    body = client.get("/videos/p2/thumbnail-plan").json()
    assert body["status"] == "withdrawn" and body["segments"] == []
    assert body["generation"] > gen

    # a second order fully inside [2,6): playable set unchanged -> plan kept
    client.post("/videos/p2/thumbnail-plan", json={"thumbnail_count": 2})
    body = _wait_plan(client, "p2")
    assert body["status"] == "published"
    kept = body
    client.post("/videos/p2/restrictions", json={"order_id": "B", "start": 3, "end": 5})
    body = client.get("/videos/p2/thumbnail-plan").json()
    assert body["status"] == "published" and body["generation"] == kept["generation"]
    assert body["segments"] == kept["segments"]

    # revoking B changes the playable set?  [2,6) still hidden by A -> no
    client.delete("/videos/p2/restrictions/B")
    body = client.get("/videos/p2/thumbnail-plan").json()
    assert body["status"] == "published"
    # revoking A re-playes [2,6) -> withdrawn
    client.delete("/videos/p2/restrictions/A")
    body = client.get("/videos/p2/thumbnail-plan").json()
    assert body["status"] == "withdrawn"


def test_plan_api_reimport_withdraws_and_blocks_stale(client):
    client.post("/videos/p3/shots", json={"shots": _shots(200, seed=13)})
    client.post("/videos/p3/thumbnail-plan", json={"thumbnail_count": 8})
    # re-import immediately, while the old solve is still running
    client.post("/videos/p3/shots", json={"shots": _shots(199, seed=14)})
    time.sleep(0.3)
    body = client.get("/videos/p3/thumbnail-plan").json()
    assert body["status"] == "none" and body["segments"] == []
    # new plan reflects the new shot count
    client.post("/videos/p3/thumbnail-plan", json={"thumbnail_count": 2})
    body = _wait_plan(client, "p3")
    assert body["status"] == "published"
    assert body["segments"][-1]["end"] == 199


def test_plan_api_infeasible_keeps_previous(client):
    shots = _shots(6, seed=15)
    client.post("/videos/p4/shots", json={"shots": shots})
    # one restricted shot splits the playable set into runs [0,2) and [3,6)
    client.post("/videos/p4/restrictions", json={"order_id": "X", "start": 2, "end": 3})
    client.post("/videos/p4/thumbnail-plan", json={"thumbnail_count": 2})
    good = _wait_plan(client, "p4")
    assert good["status"] == "published" and len(good["segments"]) == 2
    # Same playable set, K=1 cannot cover both runs -> infeasible.  No order
    # changed, so the previously published plan must survive untouched.
    client.post("/videos/p4/thumbnail-plan", json={"thumbnail_count": 1})
    body = _wait_plan(client, "p4")
    assert body["status"] == "published" and body["last_outcome"] == "infeasible"
    assert body["shots"] == good["shots"] and body["segments"] == good["segments"]

    # a request that is infeasible from the start reports infeasible, no plan
    client.post("/videos/p4b/shots", json={"shots": shots})
    client.post("/videos/p4b/restrictions", json={"order_id": "X", "start": 2, "end": 3})
    client.post("/videos/p4b/thumbnail-plan", json={"thumbnail_count": 1})
    body = _wait_plan(client, "p4b")
    assert body["status"] == "infeasible" and body["segments"] == []


def test_plan_api_segments_partition_exactly_the_playable_set(client):
    client.post("/videos/p7/shots", json={"shots": _shots(60, seed=18)})
    # restrictions [5,9) and [30,31) leave runs [0,5) [9,30) [31,60)
    client.post("/videos/p7/restrictions", json={"order_id": "A", "start": 5, "end": 9})
    client.post("/videos/p7/restrictions", json={"order_id": "B", "start": 30, "end": 31})
    visible = [i for i in range(60) if not (5 <= i < 9 or i == 30)]
    client.post("/videos/p7/thumbnail-plan",
                json={"thumbnail_count": 5, "required_shots": [visible[0], visible[-1]]})
    body = _wait_plan(client, "p7")
    assert body["status"] == "published" and len(body["segments"]) == 5
    covered = []
    for seg in body["segments"]:
        assert seg["start"] <= seg["shot"] < seg["end"]
        covered.extend(range(seg["start"], seg["end"]))
    assert covered == visible  # every playable shot, exactly once, no hidden ones
    assert set(body["required_shots"]) <= set(body["shots"])


def test_plan_api_new_request_supersedes_old(client):
    client.post("/videos/p6/shots", json={"shots": _shots(120, seed=17)})
    client.post("/videos/p6/thumbnail-plan", json={"thumbnail_count": 8})
    client.post("/videos/p6/thumbnail-plan", json={"thumbnail_count": 2})
    body = _wait_plan(client, "p6")
    assert body["status"] == "published" and body["thumbnail_count"] == 2
    assert len(body["segments"]) == 2


def test_plan_api_validation(client):
    client.post("/videos/p5/shots", json={"shots": _shots(10, seed=16)})
    assert client.post("/videos/p5/thumbnail-plan",
                       json={"thumbnail_count": 0}).status_code == 422
    assert client.post("/videos/p5/thumbnail-plan",
                       json={"thumbnail_count": 9}).status_code == 422
    assert client.post("/videos/p5/thumbnail-plan",
                       json={"thumbnail_count": 2, "required_shots": [1, 2, 3]}).status_code == 422
    assert client.post("/videos/p5/thumbnail-plan",
                       json={"thumbnail_count": 2, "required_shots": [10]}).status_code == 400
    assert client.post("/videos/p5/thumbnail-plan",
                       json={"thumbnail_count": 2, "required_shots": [1, 1]}).status_code == 400
    assert client.get("/videos/nope/thumbnail-plan").status_code == 404
