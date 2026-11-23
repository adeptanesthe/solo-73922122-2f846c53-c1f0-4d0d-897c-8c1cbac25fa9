"""Video retrieval-vector service.

Import up to 100k time-ordered shots (16-dim feature in [-1,1], attention
score in [-1000,1000]) per video, hide shot ranges by order id, and always
get back the retrieval vector computed over the currently playable shots:

    softmax(attention) weighted sum of features.

Ranges are 0-based, left-closed right-open [start, end).  Orders may
overlap; a shot stays hidden until every order covering it is revoked.
Re-registering the same order id with the same range is a no-op; the same
id with a different range is rejected with 409.  When every shot is hidden
the response carries status="withdrawn" (vector=null) as the
index-withdrawal marker.

Thumbnail plans (a separate feature) summarise all playable shots with
K = 1..8 thumbnails in playback order; see app/thumbnails.py and the
/videos/{video_id}/thumbnail-plan endpoints below.
"""

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .segment_tree import DIM, VisibleSoftmaxTree
from .thumbnails import MAX_PLAN_SHOTS, THUMB_MAX, THUMB_MIN, optimal_plan

MAX_SHOTS = 100_000
SCORE_MIN, SCORE_MAX = -1000.0, 1000.0

app = FastAPI(title="shot-retrieval-vector", version="1.0.0")


# ---------------------------------------------------------------- schemas

class Shot(BaseModel):
    feature: list[float] = Field(min_length=DIM, max_length=DIM)
    score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)

    @field_validator("feature")
    @classmethod
    def _feature_range(cls, v):
        for x in v:
            if not (-1.0 <= x <= 1.0):
                raise ValueError("feature dimensions must be within [-1, 1]")
        return v


class ImportShotsRequest(BaseModel):
    shots: list[Shot] = Field(min_length=1, max_length=MAX_SHOTS)


class RestrictionRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=128)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class VectorResponse(BaseModel):
    video_id: str
    shot_count: int
    visible_count: int
    status: str  # "ok" | "withdrawn"
    vector: list[float] | None


class PlanRequest(BaseModel):
    thumbnail_count: int = Field(ge=THUMB_MIN, le=THUMB_MAX)
    required_shots: list[int] = Field(default_factory=list, max_length=2)


class PlanSegment(BaseModel):
    start: int   # inclusive, 0-based
    end: int     # exclusive, 0-based
    shot: int    # selected thumbnail shot number


class PlanResponse(BaseModel):
    video_id: str
    thumbnail_count: int
    required_shots: list[int]
    shots: list[int]          # selected shot numbers, playback order
    segments: list[PlanSegment]
    status: str               # none|computing|published|withdrawn|infeasible|error
    generation: int           # bumped whenever the plan is withdrawn
    pending: bool = False     # a computation is currently running
    last_outcome: str = "none"  # none|pending|published|infeasible|withdrawn|error


# ----------------------------------------------------------------- state

class _Plan:
    __slots__ = ("generation", "thumbnails", "required", "segments", "shots")

    def __init__(self, generation, thumbnails, required, segments, shots):
        self.generation = generation
        self.thumbnails = thumbnails
        self.required = required
        self.segments = segments
        self.shots = shots


class _VideoState:
    __slots__ = ("tree", "orders", "features", "plan_generation", "plan",
                 "pending", "worker", "visible_snapshot", "last_outcome")

    def __init__(self, tree, features):
        self.tree = tree
        self.features = features      # per-shot 16-dim features (kept for plans)
        self.orders = {}              # order_id -> (start, end)
        self.plan_generation = 0      # bumped whenever the plan is withdrawn
        self.plan = None              # published _Plan, if any
        self.pending = None           # (token, thumbnail_count, required)
        self.worker = None            # running threading.Thread, if any
        self.last_outcome = "none"    # none|pending|published|infeasible|withdrawn
        # Playable-set snapshot, used to tell "restriction order changed but
        # the playable set did not" from a real visibility change.  Only
        # maintained for videos small enough to support thumbnail plans.
        self.visible_snapshot = (
            tree.visible_indices() if tree.n <= MAX_PLAN_SHOTS else None
        )

    def withdraw_plan(self):
        """Drop the published plan and stop any in-flight result from being
        published.  Called when the playable set actually changes."""
        self.plan = None
        self.pending = None
        self.last_outcome = "withdrawn"
        self.plan_generation += 1


_VIDEOS: dict[str, _VideoState] = {}
_LOCK = threading.Lock()          # mutations are O(log N); held only for them


def _refresh_visibility(st: _VideoState):
    """Withdraw the thumbnail plan iff the playable set actually changed.

    Must be called under _LOCK after every restriction mutation.  For videos
    above the plan shot cap no plan can exist, so the O(N) snapshot is skipped
    and the restriction path keeps its O(log N) cost.
    """
    if st.tree.n > MAX_PLAN_SHOTS:
        return
    vis = st.tree.visible_indices()
    if st.visible_snapshot is not None and vis != st.visible_snapshot:
        st.withdraw_plan()
    st.visible_snapshot = vis


def _get_state(video_id: str) -> _VideoState:
    st = _VIDEOS.get(video_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"unknown video_id: {video_id!r}")
    return st


def _plan_response(video_id, st, status=None):
    plan = st.plan
    pending = st.pending is not None
    if plan is not None:
        return PlanResponse(
            video_id=video_id,
            thumbnail_count=plan.thumbnails,
            required_shots=plan.required,
            shots=plan.shots,
            segments=[PlanSegment(**seg) for seg in plan.segments],
            status=status or "published",
            generation=plan.generation,
            pending=pending,
            last_outcome=st.last_outcome,
        )
    # No published plan.  Report the last outcome (idle, computing, or the
    # last request was infeasible/discarded) without leaking solver state.
    if status is None:
        if pending:
            status = "computing"
        elif st.last_outcome in ("infeasible", "withdrawn", "error"):
            status = st.last_outcome
        else:
            status = "none"
    return PlanResponse(
        video_id=video_id,
        thumbnail_count=(st.pending[1] if pending else 0),
        required_shots=(st.pending[2] if pending else []),
        shots=[],
        segments=[],
        status=status,
        generation=st.plan_generation,
        pending=pending,
        last_outcome=st.last_outcome,
    )


def _run_plan(video_id, st):
    """Background solver loop, at most one thread per video.

    Each iteration snapshots the current pending request and the playable
    set under the lock, solves outside it, then publishes only if the pending
    token is unchanged (not superseded, withdrawn by an import, or invalidated
    by a visibility change).  A newer request simply replaces the pending
    token and is picked up by the next iteration.
    """
    while True:
        with _LOCK:
            if _VIDEOS.get(video_id) is not st:
                return  # re-imported: this worker belongs to the old state
            if st.pending is None:
                st.worker = None
                return
            token, thumbnails, required = st.pending
            visible = tuple(st.tree.visible_indices())
            features = st.features
        try:
            result = optimal_plan(features, visible, thumbnails, required)
            error = False
        except Exception:  # never let a worker crash the service silently
            result, error = None, True
        with _LOCK:
            if _VIDEOS.get(video_id) is not st:
                return  # re-imported while computing
            if st.pending is None or st.pending[0] is not token:
                continue  # superseded or withdrawn: re-read the request
            st.pending = None
            st.worker = None
            if error:
                st.last_outcome = "error"
                return
            if result is None:
                st.last_outcome = "infeasible"  # keep the currently published plan
                return
            # Belt and braces: _refresh_visibility also drops `pending`, so a
            # changed playable set cannot reach here; compare anyway.
            if tuple(st.tree.visible_indices()) != visible:
                st.last_outcome = "withdrawn"
                return
            st.plan = _Plan(
                generation=st.plan_generation,
                thumbnails=thumbnails,
                required=list(required),
                segments=result.segments,
                shots=result.thumbnail_shots(),
            )
            st.last_outcome = "published"
            return


def _response(video_id: str, st: _VideoState) -> VectorResponse:
    vec = st.tree.retrieval_vector()
    return VectorResponse(
        video_id=video_id,
        shot_count=st.tree.n,
        visible_count=st.tree.visible_count(),
        status="ok" if vec is not None else "withdrawn",
        vector=vec,
    )


# -------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "up"}


@app.post("/videos/{video_id}/shots", response_model=VectorResponse)
def import_shots(video_id: str, req: ImportShotsRequest):
    """(Re)import a video's shots.  Re-importing replaces all shots and
    drops every existing restriction order of that video.

    It also withdraws the video's thumbnail plan and invalidates any
    in-flight plan computation, so a stale result can never be published.
    """
    features = [s.feature for s in req.shots]
    scores = [s.score for s in req.shots]
    tree = VisibleSoftmaxTree(features, scores)  # O(N), built outside the lock
    with _LOCK:
        _VIDEOS[video_id] = _VideoState(tree, features)
        return _response(video_id, _VIDEOS[video_id])


@app.post("/videos/{video_id}/restrictions", response_model=VectorResponse)
def register_restriction(video_id: str, req: RestrictionRequest):
    """Hide shots [start, end) under `order_id`.  Idempotent per
    (order_id, start, end); same id with a different range -> 409."""
    with _LOCK:
        st = _get_state(video_id)
        if req.start >= req.end or req.end > st.tree.n:
            raise HTTPException(
                status_code=400,
                detail=f"invalid range [{req.start}, {req.end}) for {st.tree.n} shots",
            )
        existing = st.orders.get(req.order_id)
        if existing is not None:
            if existing == (req.start, req.end):
                return _response(video_id, st)  # duplicate registration: no state change
            raise HTTPException(
                status_code=409,
                detail=(
                    f"order {req.order_id!r} already registered with range "
                    f"[{existing[0]}, {existing[1]}); revoke it first"
                ),
            )
        st.tree.add_cover(req.start, req.end, +1)
        st.orders[req.order_id] = (req.start, req.end)
        _refresh_visibility(st)
        return _response(video_id, st)


@app.delete("/videos/{video_id}/restrictions/{order_id}", response_model=VectorResponse)
def revoke_restriction(video_id: str, order_id: str):
    """Revoke one order.  Shots still covered by other orders stay hidden."""
    with _LOCK:
        st = _get_state(video_id)
        rng = st.orders.pop(order_id, None)
        if rng is None:
            raise HTTPException(status_code=404, detail=f"unknown order_id: {order_id!r}")
        st.tree.add_cover(rng[0], rng[1], -1)
        _refresh_visibility(st)
        return _response(video_id, st)


@app.get("/videos/{video_id}/retrieval-vector", response_model=VectorResponse)
def get_retrieval_vector(video_id: str):
    with _LOCK:
        return _response(video_id, _get_state(video_id))


# ------------------------------------------------------ thumbnail plans

def _plan_worker_start(video_id, st, req):
    """Queue a request and make sure one background solver thread is running."""
    token = object()
    st.pending = (token, req.thumbnail_count, tuple(sorted(set(req.required_shots))))
    st.last_outcome = "pending"
    if st.worker is not None:
        return  # the running solve will pick up the new pending request
    worker = threading.Thread(target=_run_plan, args=(video_id, st),
                              name=f"plan-{video_id}", daemon=True)
    st.worker = worker
    worker.start()


@app.post("/videos/{video_id}/thumbnail-plan", response_model=PlanResponse)
def compute_thumbnail_plan(video_id: str, req: PlanRequest):
    """Start computing an optimal K-thumbnail plan for the playable shots.

    Returns immediately with status="computing"; poll GET for the result.
    Requires the import + restriction endpoints to stay responsive, so the
    solve runs on a background thread.  If a plan was already published it is
    withdrawn when the playable set changes (not when only the restriction
    orders change); an infeasible request keeps the previous plan.
    """
    with _LOCK:
        st = _get_state(video_id)
        if st.tree.n > MAX_PLAN_SHOTS:
            raise HTTPException(
                status_code=400,
                detail=f"thumbnail plans support at most {MAX_PLAN_SHOTS} shots, "
                       f"this video has {st.tree.n}",
            )
        for shot in req.required_shots:
            if not (0 <= shot < st.tree.n):
                raise HTTPException(
                    status_code=400,
                    detail=f"required shot {shot} out of range [0, {st.tree.n})",
                )
        if len(set(req.required_shots)) != len(req.required_shots):
            raise HTTPException(status_code=400,
                                detail="required_shots must be distinct")
        _plan_worker_start(video_id, st, req)
        return _plan_response(video_id, st, status="computing")


@app.get("/videos/{video_id}/thumbnail-plan", response_model=PlanResponse)
def get_thumbnail_plan(video_id: str):
    """Return the published plan (its segments' half-open intervals and
    selected shot numbers), or the computation state."""
    with _LOCK:
        return _plan_response(video_id, _get_state(video_id))
