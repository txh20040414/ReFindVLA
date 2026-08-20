# ReFindVLA Code

本目录包含当前 ReFindVLA 的两条可复现代码链：

1. `main_interactive_track/`：在线闭环控制。它在本地连接 CARLA-Air/AirSim，向 Linux GPU 上的 Qwen2.5-VL + LoRA 服务发送图像和上下文，接收高层结构化决策，再由本地航点规划器和安全控制器生成 AirSim 控制量。
2. `collect_data_code/`：决策级数据采集和数据检查。它不调用 Qwen 推理服务，只采集图像、状态、目标框和场景元数据。

## 在线闭环入口

```bash
cd /path/to/ReFindVLA/Carla_code/Code
python3 -m main_interactive_track.main
```

运行前需要：

- CARLA-Air 已启动并监听 CARLA `2000` 端口；
- AirSim 接口可访问 `41451` 端口；
- Linux GPU 机器上的 `model_server.py` 已启动并通过 `SERVER_URL` 可访问；
- 如果 `carla` Python 包没有安装到当前环境，设置 `CARLA_AIR_ROOT` 指向 CARLA-Air 安装目录；该目录下应包含 `PythonAPI/`。

例如：

```bash
export CARLA_AIR_ROOT=/path/to/CarlaAir-v0.1.7
export SERVER_URL=http://GPU_HOST:8000
export TRACKING_LOG_DIR=/path/to/outputs/decision_runs
export RECOVER_USE_REMOTE_VLA=1
```

当前在线服务输出的是高层 JSON：

```json
{
  "phase": "search",
  "view": "top",
  "target_found": false,
  "confidence": 0.45,
  "hold_seconds": 5,
  "reason": "沿最后方向搜索"
}
```

模型不直接输出 `vx/vy/vz/yaw_rate`。这些低层控制量由本地 `decision.py`、`controller.py` 和 AirSim 接口计算。旧版持续跟踪链及其低层数据转换、旧版跟踪评估入口已经移除。

设置 `RECOVER_USE_REMOTE_VLA=0` 可运行保留记忆、信念图、候选验证和控制器的本地规则模式，用于隔离 VLA 决策模块。该模式只提供实验代码开关，不自动产生或声称已有消融结果。

## 当前模块

- `main_interactive_track/main.py`：在线入口；
- `recover_vla.py`：异步 VLA 请求、观测、决策和控制循环；
- `simulator.py`：CARLA/AirSim 连接、车辆生成、坐标转换和图像投影；
- `observation.py`：候选车辆观测和目标记忆门控；
- `target_memory.py`、`belief_map.py`：历史线索和目标搜索区域；
- `candidate_verification.py`：可解释的多线索候选排序与确认门控；
- `environment_context.py`：道路分支和建筑安全上下文；
- `decision.py`：结构化决策解析、语义航点和本地兜底；
- `controller.py`：语义航点到 AirSim body-frame 控制量；
- `logger.py`：决策、控制和帧日志；
- `model_server.py`：Qwen2.5-VL + ReFindVLA LoRA HTTP 服务。

## 决策级数据采集

```bash
cd /path/to/ReFindVLA/Carla_code/Code
python3 -m collect_data_code.record_find_track_data \
  --output-root /path/to/outputs/find_track_data \
  --episodes 1
```

采集脚本保存图像、JSON sidecar、YOLO 归一化框和采集配置。它使用 CARLA 投影框生成仿真标注，不代表真实检测器或真实 ReID 的性能。

## 参数和实验协议

所有运行参数都可以通过环境变量覆盖，主要参数定义在 `main_interactive_track/config.py`。特别是确认阈值、检查阈值、记忆更新阈值和候选间隔必须与所报告的实验批次保持一致。历史实验记录和当前代码默认值不是同一版本时，必须在实验日志中明确标注，不能混用。

本轮返修实验文档：

- [`docs/EXPERIMENT_DESIGN_ROUTE.md`](../docs/EXPERIMENT_DESIGN_ROUTE.md)：完整系统、严格 no-VLA、A* 和螺旋搜索的统一实验路线；
- [`docs/REVIEW_ISSUES_EVIDENCE_MATRIX.md`](../docs/REVIEW_ISSUES_EVIDENCE_MATRIX.md)：审稿问题、代码证据、论文处理和实验状态矩阵；
- [`docs/EXPERIMENT_LOG_TEMPLATE.md`](../docs/EXPERIMENT_LOG_TEMPLATE.md)：批次和 episode 原始记录模板；
- [`docs/LINUX_GPU_VLA_EXPERIMENT.md`](../docs/LINUX_GPU_VLA_EXPERIMENT.md)：本地 CARLA-Air 与远程 Linux Qwen 服务的操作手册。
