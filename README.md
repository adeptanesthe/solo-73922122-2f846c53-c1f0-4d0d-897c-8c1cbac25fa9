# 片库镜头检索向量服务

用可播镜头实时维护视频检索向量：导入按时间排序的 1～100,000 个镜头（每个带 16 维特征、注意力分数），
按单号登记/撤销镜头隐藏区间，每次变更后以 **O(log N)** 返回由可播镜头算出的检索向量
（注意力分数 softmax 加权特征和）。

## 运行

```bash
# Docker Compose（推荐）
docker compose up --build

# 或本地运行
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务监听 `:8000`，健康检查 `GET /health`，交互文档 `GET /docs`。

演示脚本（重叠区间的登记与撤销，含幂等重送、冲突、全隐藏撤下、恢复）：

```bash
bash scripts/demo.sh            # BASE=http://localhost:8000 可覆盖
```

缩略图方案演示（发布 → 可播集合改变而撤下 → 限播单变化但可播集合不变而保留）：

```bash
bash scripts/demo_plan.sh       # BASE=http://localhost:8000 可覆盖
```

等价的原始请求（发布；随后登记限播单导致撤下）：

```bash
# 1) 导入 10 个镜头（scripts/demo_shots.json）
curl -s -X POST localhost:8000/videos/demo/shots \
  -H 'Content-Type: application/json' --data @scripts/demo_shots.json
# 2) 发布：3 张缩略图，指定镜头 1、7（异步，立即返回 computing）
curl -s -X POST localhost:8000/videos/demo/thumbnail-plan \
  -H 'Content-Type: application/json' \
  -d '{"thumbnail_count":3,"required_shots":[1,7]}'
# 3) 轮询 -> published
curl -s localhost:8000/videos/demo/thumbnail-plan
# 4) 登记限播 -> 可播集合改变 -> 方案撤下
curl -s -X POST localhost:8000/videos/demo/restrictions \
  -H 'Content-Type: application/json' \
  -d '{"order_id":"lic-A","start":2,"end":6}'
curl -s localhost:8000/videos/demo/thumbnail-plan
# {"status":"withdrawn","segments":[],"generation":1,...}
```

自测：

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/videos/{video_id}/shots` | 导入镜头。body: `{"shots":[{"feature":[16 个 −1~1],"score":−1000~1000}, ...]}`，1～100,000 条。重复导入整体替换并清空该视频所有订单 |
| `POST` | `/videos/{video_id}/restrictions` | 登记隐藏单。body: `{"order_id":"lic-A","start":2,"end":6}`，区间 0 起算、左闭右开 |
| `DELETE` | `/videos/{video_id}/restrictions/{order_id}` | 撤销一单；仍被其他单覆盖的镜头保持隐藏 |
| `GET` | `/videos/{video_id}/retrieval-vector` | 查询当前检索向量 |
| `POST` | `/videos/{video_id}/thumbnail-plan` | 发起缩略图方案计算（异步）。body: `{"thumbnail_count":1~8,"required_shots":[最多 2 个镜头序号]}`，镜头数不超过 200 |
| `GET` | `/videos/{video_id}/thumbnail-plan` | 查询已发布方案（每段左闭右开区间 + 所选镜头序号，均 0 起算）或计算状态 |
| `GET` | `/health` | 健康检查 |

统一响应：

```json
{"video_id":"demo","shot_count":10,"visible_count":6,"status":"ok","vector":[...16 维...]}
```

- **幂等**：同 `order_id` 同范围重送 → 200，状态不变。
- **冲突**：同 `order_id` 不同范围 → 409（提示先撤销）。
- **全隐藏**：`status="withdrawn"`、`vector=null` —— 即撤下索引的标记；撤销后自动恢复。
- 校验失败 422（维度数、特征/分数越界、镜头数越界），区间语义错误 400（`start>=end`、`end>N`），未知视频/单号 404。

### 缩略图方案

用 1～8 张按播放顺序排列的缩略图概括整片**可播**内容。把可播镜头（按播放顺序）恰好切成同样多的
连续段，每段选一个镜头作缩略图；任何一段都不能跨过限播镜头（即一段只能落在一段连续可播镜头内）。
指定镜头（最多 2 个）必须入选。代价为每个可播镜头到本段所选镜头特征的**平方欧氏距离之和**，取最小；
同分时取缩略图序号列表**字典序最小**的一组。

请求立即返回 `status="computing"`，求解在后台线程进行，`POST /shots`、登记/撤单等接口照常可用；
轮询 `GET` 得到最终结果：

```bash
curl -X POST localhost:8000/videos/demo/thumbnail-plan \
  -H 'Content-Type: application/json' \
  -d '{"thumbnail_count":3,"required_shots":[1,7]}'
# {"status":"computing","pending":true,...}

curl localhost:8000/videos/demo/thumbnail-plan
# {"video_id":"demo","thumbnail_count":3,"required_shots":[1,7],
#  "shots":[0,1,7],
#  "segments":[{"start":0,"end":1,"shot":0},{"start":1,"end":7,"shot":1},
#              {"start":7,"end":10,"shot":7}],
#  "status":"published","generation":0,"pending":false,"last_outcome":"published"}
```

`status`：`none`（从未请求）/ `computing`（首次计算中，POST 的返回值）/ `published` /
`withdrawn`（可播集合改变后撤下）/ `infeasible`（无解且无旧方案）/ `error`（求解异常）。
若已有旧方案而新一轮计算正在进行，`GET` 仍返回 `status="published"`、`pending=true`，数据是旧方案，直到新结果发布（或可播集合改变而被撤下）；
`POST` 则直接返回 `status="computing"`（不附带旧方案数据，旧方案仍可经 `GET` 查到，直到新结果发布或被撤下）。
每段 `start`/`end` 为 0 起算左闭右开区间，`shot` 为该段所选镜头序号；区间内只含可播镜头，
不同的段之间可以直接存在被限播镜头造成的空档。

生命周期：

- **无解**（如 K 小于连续可播段数、指定镜头被限播、指定镜头数超过 K）：不发布新结果，`GET` 仍返回
  原方案（`last_outcome="infeasible"`）；从未发布过则 `status="infeasible"`、`segments=[]`。
- **重新导入**：撤下原方案，并作废正在计算的旧结果（旧结果不会发布）。
- **可播集合实际改变**（登记/撤销限播单后真正增加了/减少了可播镜头）：撤下原方案、作废在算结果，
  `generation` 自增。
- **限播单变化但可播集合不变**（例如新单完全落在已隐藏区间内、或撤销后仍被其他单覆盖）：原方案仍可查。
- 计算期间可照常导入、接单、撤单；可播集合一变，在算的旧结果自动作废，不会发布。
- 镜头数超过 200 的视频不支持缩略图方案（400）。

## 实现（`app/segment_tree.py`）

线段树维护可播集合上的稳定 softmax 加权和。每个节点存：

- `cnt`：可播镜头数
- `mx`：可播注意力分数最大值
- `sm = Σ exp(score − mx)`
- `vec[16] = Σ exp(score − mx) · feature`

兄弟合并时平移到共同最大值（经典 online-softmax 分解）：

```
m   = max(mx_l, mx_r)
sm  = sm_l·exp(mx_l−m) + sm_r·exp(mx_r−m)
vec = vec_l·exp(mx_l−m) + vec_r·exp(mx_r−m)
```

隐藏用**懒引用计数**：登记对区间覆盖的 O(log N) 个规范节点 `cover += 1`，撤销 `cover -= 1`；
`cover > 0` 的节点贡献为空。多单重叠时同一镜头计数 ≥1，撤销一单后若仍被其他单覆盖则保持隐藏。
子节点聚合始终有效，无需下推。检索向量 = 根 `vec / sm`（逐维）。

**复杂度**：登记/撤销各 O(log N) 次节点访问（每次合并为 16 维常数工作）；导入建树 O(N)；查询 O(1)。
N=100,000 时树约 2¹⁸ 个节点，扁平 `array('d')` 存储，内存约 40 MB/视频。

**精度**：合并只做 max 平移与浮点乘加，与"稳定全量计算"（max 平移 + `math.fsum` 精确求和的参考实现）
的差异仅来自求和结合顺序，误差随树高 ~log₂N 线性累积，理论量级 ~1e-15，远小于 1e-9 的验收阈值。
分数区间 [−1000, 1000] 内 `exp` 不会上溢，下溢到 0 与全量计算行为一致。

## 缩略图方案实现（`app/thumbnails.py`）

设可播位置为 0…M−1，`cost(p,j,q)` 为把位置 `[p,j)` 归为一段、选 q 作缩略图的距离和。
标准连续分段 DP：

```
dp[g][j] = min_{p<j, q∈[p,j)} dp[g-1][p] + cost(p,j,q)
```

- 段内若有指定镜头，则 q 只能取该镜头（一段最多容纳一个指定镜头，两个指定镜头挤进一段即视为不可行）。
- 段必须连续可播（位置 `p..j-1` 对应的镜头序号连续），否则该段直接不可选。
- 先按缩略图 q 预计算距离前缀和 `pref[q][j] = Σ_{i<j} ||f_i − f_q||²`：O(M²) 次距离、O(M²) 存储；
  枚举 (p, j) 取段最优 q 为 O(M³)（M ≤ 200，约 1.3M 次 O(1) 查表），DP 本身 O(K·M²)。
  **不做分段穷举**（穷举是 C(M−1,K−1) 量级）。
- 同分精度：把每个浮点特征按同一个 2 的幂缩放成**精确整数**（IEEE-754 双精度都是二进制有理数），
  因而距离和及其前缀和都是精确整数、相等代价逐位相等，不使用 epsilon；字典序比较也精确。
- 状态比较键为 (代价, 缩略图序号元组)：代价为精确整数、序号元组按字典序比较，因此最优性与同分
  规则都精确成立；每段先只保留同代价下最小的 q，再由 DP 在等代价时取字典序最小的序号元组。
- 求解为 O(K·M² + M³) 时间、O(M²) 内存，且在循环里周期性 `sleep(0)` 让出 GIL，
  保证后台求解时其他请求仍有响应。

## 实测结果（本机 Python 3.11，pytest 9.2）

- **82 个测试全部通过**（`tests/test_service.py`）：
  - 40 组随机用例（N=1…1000，每步登记/撤销后比对参考实现）：逐维误差 ≤ 1e-9 ✅
  - 极端值（分数 ±1000、特征 ±1）✅；N=100,000 大样本 ✅；全隐藏→恢复 ✅
  - API：重叠撤销、幂等重送、409 冲突、404、422/400 校验、withdrawn 标记 ✅
  - 缩略图方案：25 组随机用例与**穷举**最优解逐一比对（含限播分段、指定镜头、字典序同分）✅；
    另以 4000 组“低取值/重复特征”抗同分用例复核主导同分分支 ✅
  - 缩略图 API：异步计算期间导入/接单/撤单可用、可播集合不变时方案保留、改变时撤下、
    重新导入作废旧计算、无解保留原方案、新请求覆盖旧请求、段恰好覆盖全部可播镜头、参数校验 ✅
- **缩略图性能实测**：M=200、K=8 求解约 0.8～1.1 s；同分密集输入 0.8 s、极端次正规特征 2.4 s；
  计算期间其他请求延迟约 40～100 ms（对照计算前 ~5 ms）。
- **精度实测**：N=100,000 下 60 个随机隐藏状态，逐维最大误差 **8.3e-17**（阈值 1e-9）。
- **复杂度实测**（2000 次登记 + 2000 次撤销取平均）：
  - N=10,000 → 89.7 µs/次；N=100,000 → 181.5 µs/次。N 涨 10 倍耗时仅 ~2 倍，符合 O(log N)。
- **端到端（HTTP）**：导入 100,000 镜头（16 MB body）3.7 s；登记/撤销单次 1.5～10 ms；
  撤销重叠单后 `visible_count` 正确；全隐藏返回 `{"status":"withdrawn","vector":null}`。
- **演示**：`scripts/demo.sh` 输出见上节描述，撤销 lic-A 后 [4,8) 仍被 lic-B 隐藏，
  全部撤销后向量与导入时逐位一致。
- **缩略图演示（HTTP 实测）**：`scripts/demo_plan.sh` —— 10 镜头、K=3、指定镜头 1/7，发布为
  `[{0,1),[1,7),[7,10)}` 选 `[0,1,7]`；登记 lic-A [2,6) 后可播集合改变，`status="withdrawn"`；
  重发布为 `[{0,2),[6,10)}` 选 `[0,9]`；再登记完全落在隐藏区的 lic-B [3,5)，方案仍 `published`
  且 `generation` 不变；撤销 lic-A 后重新 `withdrawn`。
- **并发压测（本机）**：导入线程（4 次整体替换）+ 限播线程（40 次登记/撤销）+ 方案线程
  （6 次连续请求）并发运行，无 5xx；结束后无残留工作线程，最终状态与当前可播集合一致。

## 未完成项 / 限制

1. **Docker 未实测**：本环境无 Docker 守护进程，`Dockerfile` / `docker-compose.yml` 未实际构建运行
  （已做 YAML 语法校验；镜像为标准 `python:3.11-slim` + `requirements.txt`，风险低）。
   缩略图方案可在 `docker compose up --build` 后用 `scripts/demo_plan.sh` 复核。
2. **单进程内存态**：数据存于进程内存，重启即失；多 worker/多副本不共享状态（compose 固定单容器单 worker）。
   生产化需外置存储或分片路由。
3. **导入耗时**：100,000 镜头导入约 2～4 s（O(N) 建树 + JSON 校验），导入期间会阻塞其他请求
  （FastAPI 线程池内持锁仅覆盖换指针，瓶颈在 CPU 本身）。
4. 重复导入同一 `video_id` 会**清空其全部订单**（按"整体替换"语义实现，如需保留可改）。
5. 无鉴权、无持久化、无速率限制；`order_id` 仅在单个视频内唯一。
6. **缩略图求解隔离**：后台求解靠 `sleep(0)` 协作让出 GIL，未用独立进程；单核/高负载下仍会
   抬高其他请求延迟（实测 40～100 ms）。每个视频同一时刻只保留一个在算任务，新请求会作废旧任务。
7. **可播集合变化判定**：限播单变更后对比可播快照（O(M) 遍历树，仅对 ≤200 镜头视频进行，
   因此不影响大视频 O(log N) 的接单/撤单）。
8. **缩略图方案仅内存**：不持久化，随重启/重新导入消失；不支持跨视频批量求解。
