# RecoverVLA 算法与接口说明

## 目标

本目录实现的是 RecoverVLA：语言意图驱动的无人机动态目标找回系统。它不是单纯追车，而是在目标短时丢失、建筑遮挡、岔路不确定、相似车辆干扰等场景中，根据目标最后一次出现的位置、速度和方向预测搜索区域，让无人机主动飞往更可能重新发现目标的位置，并通过候选确认降低错认。

## 主入口

```bash
cd /path/to/ReFindVLA/Carla_code/Code
export SERVER_URL=http://GPU_SERVER_IP:8000
python3 -m main_interactive_track.main
```

如果只想检查本地逻辑，不强制连接远程模型：

```bash
export RECOVER_USE_REMOTE_VLA=0
export RECOVER_REMOTE_REQUIRED=0
python3 -m main_interactive_track.main
```

## 模块结构

- `main.py`：入口，加载 `recover_vla.main`。
- `recover_vla.py`：交互式主循环，连接 CARLA-Air、AirSim、远程模型服务，执行完整闭环。
- `config.py`：所有运行参数和环境变量。
- `simulator.py`：CARLA-Air/AirSim 接口适配，包括连接、起飞、图像读取、车辆生成、坐标转换、bbox 投影。
- `server_client.py`：远程 Qwen2.5-VL + LoRA 服务 HTTP 客户端。
- `target_memory.py`：目标记忆管理，记录外观、最后位置、速度、方向和丢失时间。
- `belief_map.py`：基于最后位置、速度、方向和远程延迟做目标区域预测。
- `candidate_verification.py`：候选确认，融合几何可见性、车辆属性、运动一致性、空间先验和 ReID 占位分数。
- `observation.py`：车辆候选观测与记忆门控，投影当前视野内全部车辆，只在可见且高置信确认时更新目标记忆，避免目标不可见时使用 CARLA 真值作弊。
- `environment_context.py`：CARLA 道路/建筑上下文，基于 `get_waypoint()`、`waypoint.next()` 生成 Top-K 道路分支，并通过建筑包围盒估计安全高度。
- `decision.py`：构建 VLA prompt，解析模型 JSON，生成 search/inspect/confirm/reacquire/follow/return_home 阶段计划。
- `controller.py`：低层安全控制器，把语义航点转为 AirSim body-frame 速度和 yaw_rate。
- `logger.py`：保存 JSONL、带文字和 bbox 的帧、YOLO 标签。

## 算法流程

1. 输入中文任务，例如“找回失踪的白色厢式卡车，它最后向东行驶”。
2. 系统解析目标颜色和车型，并在 CARLA 中生成或使用已有目标车。
3. `observation.py` 将当前相机中可见车辆投影为候选列表；只有可见候选通过确认阈值时，`TargetMemory` 才记录目标上一次观测到的位置、速度、方向和丢失时间。
4. 目标不可见时，系统不再读取目标 actor 真值更新记忆，而是由 `BeliefMap` 用常速度/加速度裁剪预测目标未来位置，形成 `BeliefRegion`。
5. `environment_context.py` 从 CARLA 地图中提取道路分支和建筑安全上下文，给 `BeliefRegion` 增加 Top-K road branches 和 recommended altitude。
6. `CandidateVerifier` 对当前可见候选进行综合评分：
   `Score = 0.35*几何可见性 + 0.25*车辆属性 + 0.20*运动一致性 + 0.10*空间先验 + 0.10*ReID`。当前 ReID 项为固定的中性占位分 `0.50`，不是独立 ReID 网络输出。
7. 远程 VLA 每隔数秒读取当前图像、任务、目标记忆、信念区域、道路分支、安全上下文和候选评分，输出高层阶段。
8. 本地决策器根据远程输出和本地兜底规则生成语义航点。
9. 安全控制器高频执行航点控制；建筑风险高或高度不足时先升高再水平搜索，远程推理慢时仍然沿预测区域搜索。
10. 保存 `decision.jsonl`、`frames/decision_*.png`、`frames/control_*.png`、旁路 `.json` 和 YOLO `.txt`。

## 远程模型 API

当前代码假设服务器提供：

- `GET /health`
- `POST /predict`

请求：

```json
{
  "image_b64": "PNG base64",
  "user_text": "RecoverVLA prompt"
}
```

响应推荐：

```json
{
  "response": "{\"phase\":\"search\",\"view\":\"top\",\"target_found\":false,\"confidence\":0.45,\"hold_seconds\":5,\"reason\":\"沿最后方向搜索\"}",
  "action": {
    "phase": "search",
    "view": "top",
    "target_found": false,
    "confidence": 0.45,
    "hold_seconds": 5,
    "reason": "沿最后方向搜索"
  },
  "latency_ms": 1200
}
```

当前服务器和客户端使用上面的高层结构化字段。模型不直接输出 `vx/vy/vz/yaw_rate`，这些量由本地控制器生成。

## 当前限制和后续接口

- 真实候选车辆检测/ReID 目前没有接入独立检测器，当前 `CandidateVerifier` 使用 CARLA 目标 actor 投影的 bbox 作为可解释标注。后续如果你接 YOLO/ReID，应把检测结果传入 `candidate_verification.py`。
- 当前候选列表来自 CARLA 车辆投影，可作为论文可解释 online oracle detector；后续接 YOLO/ReID 时，只需要把检测框、类别、外观 embedding 传入 `CandidateVerifier.score_candidates()`。
- 道路拓扑已使用 CARLA 最近车道 waypoint 和 `waypoint.next()` 生成局部 Top-K branch candidates；全局路网概率图可继续扩展为 episode-level graph memory。
- 电池接口在 AirSim 状态里不一定存在，当前只预留字段；真实电池返航策略需要确认 CARLA-Air/AirSim 是否提供电量。

## 论文表述对应

代码中的方法可写为：

`RecoverVLA = Target Memory + Belief Search + Candidate Verification + VLA Policy + Safety Controller`

主指标建议对应日志统计：

- Recovery Success Rate：`plan.target_found=true` 且候选确认通过的 episode 比例。
- Time-to-Reacquire：`memory.lost_since` 到首次 `phase=reacquire/follow` 的时间。
- False Reacquisition Rate：后续接入 GT candidate_id 后统计。
- Collision Rate：`simGetCollisionInfo().has_collided`。
- Search Path Length：累计无人机位置变化。
