# Windows CARLA-Air + Linux GPU Qwen 实验手册

本手册对应当前仓库的真实代码链：CARLA-Air/AirSim 在本地 Windows 运行，`main_interactive_track` 也在本地运行；Qwen2.5-VL-7B-Instruct + ReFindVLA LoRA 在远程 Linux GPU 运行，通过 HTTP `/predict` 传输图像和文本上下文。

```text
Windows 本地: CARLA-Air + CARLA 2000 + AirSim 41451 + main_interactive_track
                                      │ HTTP POST /predict
                                      ▼
Linux GPU: Code/model_server.py:8000 + Qwen2.5-VL-7B + models_/mode_v2
```

Qwen 输出高层 JSON（`phase/view/target_found/confidence/hold_seconds/reason`），不输出 `vx/vy/vz/yaw_rate`。低层速度和偏航控制只由本地 `decision.py`、`controller.py` 和 AirSim `moveByVelocityBodyFrameAsync` 执行。

## 1. 官方 CARLA-Air 接口约束

官方资料：[Quick-Start Guide](https://github.com/louiszengCN/CarlaAir/blob/main/CarlaAir_Release/guide/Quick-Start.md)、[Coordinate Systems](https://github.com/louiszengCN/CarlaAir/blob/main/CarlaAir_Release/guide/COORDINATE_SYSTEMS.md)、[Windows release](https://github.com/louiszengCN/CarlaAir/releases)。官方端口为 CARLA `2000`、AirSim `41451`。Windows 使用官方 Windows v0.1.7 二进制包，不要在 Windows 上执行 Linux 的 `./CarlaAir.sh`。

官方 FAQ 对本实验有两个关键限制：

1. Shipping 构建中 `simSetCameraPose` 可能触发 C++ abort；当前代码默认 `RECOVER_CAMERA_CONTROL=0`，使用 `%USERPROFILE%\Documents\AirSim\settings.json` 的静态相机。
2. 无人机和高密度交通同时运行时，官方建议自动驾驶车辆不超过 8 辆；当前在线代码默认 8 辆。复现旧的 10 辆批次时，必须显式设置 `RECOVER_ALLOW_UNSAFE_TRAFFIC=1` 并记录。

## 2. Windows 本地仿真器与控制端

按官方 release 下载并启动 Windows v0.1.7，在 `%USERPROFILE%\Documents\AirSim\settings.json` 配置相机、无人机和端口。当前代码默认不调用动态相机 API。

```powershell
cd C:\path\to\ReFindVLA\Carla_code
py -3 -m venv .venv-client
.\.venv-client\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-client.txt
```

```powershell
$env:CARLA_AIR_ROOT = 'C:\path\to\CarlaAir-v0.1.7-win11-x86_64'
$env:CARLA_HOST = '127.0.0.1'
$env:CARLA_PORT = '2000'
$env:AIRSIM_HOST = '127.0.0.1'
$env:SERVER_URL = 'http://GPU_SERVER_IP:8000'
$env:TRACKING_LOG_DIR = 'C:\path\to\outputs\decision_runs'
$env:RECOVER_CAMERA_CONTROL = '0'
$env:RECOVER_USE_REMOTE_VLA = '1'
$env:RECOVER_REMOTE_REQUIRED = '1'
```

`CARLA_AIR_ROOT` 可指向解压根目录；当前代码会检查 `PythonAPI`、`PythonAPI\carla` 和 `PythonAPI\carla\dist\*.egg`。默认坐标偏移是 Town10HD 的测量值：`X=172.20, Y=-183.86, Z=27.45`。它们不是跨地图常数；切换地图或 PlayerStart 时，按官方坐标文档重新校准，并设置 `CARLA_AIR_OFFSET_X/Y/Z` 与 `CARLA_AIR_OFFSET_SOURCE`。

预检和启动：

```powershell
cd C:\path\to\ReFindVLA\Carla_code
python Code\tools\validate_experiment_setup.py --server-url $env:SERVER_URL
cd Code
python -m main_interactive_track.main
```

输入例如：

```text
spawn 白色 卡车
找回失踪的白色卡车，它最后向东行驶
quit
```

`spawn` 创建目标车；当前入口是交互式单集入口，不会自动重置 CARLA 场景，也不会自动生成 E1/E2 成对 manifest。

## 3. 远程 Linux GPU Qwen 服务

```bash
cd /data/ReFindVLA/Carla_code
python3 -m venv .venv-server
source .venv-server/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-server.txt
export QWEN_BASE_MODEL=/data/models/qwen/Qwen2.5-VL-7B-Instruct
export REFINDVLA_LORA_PATH=/data/ReFindVLA/Carla_code/models_/mode_v2
test -f "$QWEN_BASE_MODEL/config.json"
test -f "$REFINDVLA_LORA_PATH/adapter_config.json"
cd /data/ReFindVLA/Carla_code/Code
python3 model_server.py
```

```bash
curl http://127.0.0.1:8000/health
nvidia-smi
```

服务绑定 `0.0.0.0:8000`；只允许 Windows 控制机访问，不要把无认证服务暴露到公网。若远程目录没有 `Code/model_server.py`，说明是旧版仓库或错误分支：

```bash
cd /data/ReFindVLA
git fetch origin
git checkout experiment-design
git pull --ff-only origin experiment-design
ls -lh Carla_code/Code/model_server.py
```

## 4. 返修实验开关和实际状态

### E0：smoke test

只验证 CARLA、AirSim、HTTP `/health`、HTTP `/predict`、控制动作和日志生成；不进入论文统计。

### E1：完整远程 VLA

```powershell
$env:RECOVER_USE_REMOTE_VLA = '1'
$env:RECOVER_REMOTE_REQUIRED = '1'
python -m main_interactive_track.main
```

已实现：远程 Qwen 请求、结构化决策解析、候选验证、Target Memory、Belief Map、道路上下文、语义航点和本地 AirSim 控制。

### E2：严格 no-VLA 对照

```powershell
$env:RECOVER_USE_REMOTE_VLA = '0'
$env:RECOVER_REMOTE_REQUIRED = '0'
python -m main_interactive_track.main
```

已实现：保留相同的记忆、信念图、候选校验、航点规划器和安全控制器，只跳过远程 Qwen，由本地规则 fallback 生成高层计划。没有同一场景 manifest 的成批运行前，不能写成论文结果。

### E3/E4：A* 和螺旋搜索

当前 `Carla_code` 没有可核验的独立 A* 或螺旋搜索运行入口，尚未完整实现。若队员提供真实基线代码，必须接入相同 manifest、控制器、终止条件和日志格式后再运行。

## 5. 日志和汇总

每次在线 episode 会创建 `TRACKING_LOG_DIR/recover_YYYYMMDD_HHMMSS/`，包含 `run_config.json`、`decision.jsonl` 和 `frames/`。配置文件保存策略、阈值、坐标偏移来源、相机模式和环境配置；JSONL 自动写入时间戳、决策、控制、延迟、候选和 `episode_summary`。成功标签是“高层计划确认且 CARLA 投影目标同时可见”，这是离线评测 oracle，不是模型输入。

```powershell
python Code\tools\aggregate_experiment_runs.py `
  --input C:\path\to\outputs\decision_runs `
  --output C:\path\to\outputs\summary_E1 `
  --label E1
```

输出 `summary.csv` 和 `summary.json`；RSR 分母是 `decision.jsonl` 文件数（episode/run 数），不是控制循环数或 Qwen 请求数。

## 6. 目前不能直接声称完成的部分

本次代码修正确保运行安全和证据记录，但以下内容仍需真实仿真补齐：

1. E1/E2 使用同一场景、同一初始状态和同一目标丢失触发条件的成对批次；当前交互入口没有自动 reset 和 loss/occlusion manifest。
2. 论文所需 RSR、TTR、路径长度、耗时、碰撞和误确认率；代码有原始日志和汇总器，但没有替用户捏造结果。
3. A*、螺旋搜索独立基线；当前仓库没有这两条可运行实现。
4. 多基座模型、真机实验和大规模外部学习基线；仍属于论文限制和未来工作。

## 7. 数据采集入口

采集脚本不调用 Qwen，只保存 CARLA 投影框、状态、图像和场景元数据：

```powershell
cd C:\path\to\ReFindVLA\Carla_code\Code
python -m collect_data_code.record_find_track_data `
  --output-root C:\path\to\outputs\find_track_data `
  --episodes 1
```

默认背景车为 8。明确复现旧批次时才使用 `--traffic 10 --allow-unsafe-traffic`。目标框来自 CARLA 几何投影，不代表独立视觉检测器或独立 ReID 模型性能。
