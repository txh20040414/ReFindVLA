# ReFindVLA Linux + GPU + CARLA-Air 实验手册

本手册对应当前仓库中的代码版本。它把运行过程分成三个部分：

```text
CARLA-Air/AirSim 仿真器
        ↓ 本地图像、状态和候选观测
Carla_code/Code/main_interactive_track
        ↓ HTTP /predict
Linux GPU 上的 Qwen2.5-VL-7B + ReFindVLA LoRA 服务
        ↓ phase/view/confidence 等高层 JSON
本地语义航点规划器 + 安全控制器
        ↓ AirSim body-frame velocity/yaw 控制
无人机在 CARLA-Air 中执行搜索、检查和找回
```

Qwen 服务输出的是高层结构化决策，不直接输出无人机底层速度。底层 `vx/vy/vz/yaw_rate` 只在本地控制器内部生成和执行。

## 1. 仓库和外部组件

本仓库只提交 ReFindVLA 的控制代码、模型服务代码、LoRA adapter 文件（如果选择提交）和实验说明，不提交完整 CARLA-Air 可执行程序，也不提交 Qwen 基座模型。

当前本地资料中使用的基础组件为：

- CARLA-Air v0.1.7；
- CARLA 0.9.16 Python API；
- AirSim 1.8.1 Python API；
- Qwen2.5-VL-7B-Instruct；
- PEFT 0.10.0；
- LoRA adapter 配置：`r=8`、`alpha=16`、`dropout=0.05`、目标层 `q_proj/k_proj/v_proj/o_proj`、`task_type=CAUSAL_LM`。

CARLA-Air 的可执行程序和 Python API 需要队员根据其许可证和官方发布方式单独安装。当前代码通过 `CARLA_AIR_ROOT` 或已经安装好的 `carla` Python 包连接它。

## 2. Linux GPU 机器：安装 Qwen 推理服务

在 Linux GPU 机器上创建独立环境：

```bash
cd /path/to/ReFindVLA/Carla_code
python3 -m venv .venv-server
source .venv-server/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-server.txt
```

Qwen 基座模型不放入仓库。将 `Qwen2.5-VL-7B-Instruct` 放到队员自己的模型目录，并设置环境变量。例如，本地训练记录中使用过：

```bash
export QWEN_BASE_MODEL=/data/models/qwen/Qwen2.5-VL-7B-Instruct
```

如果仓库中包含 `models_/mode_v2/`，它是 LoRA adapter；也可以通过环境变量指定其他已验证的 adapter：

```bash
export REFINDVLA_LORA_PATH=/path/to/ReFindVLA/Carla_code/models_/mode_v2
```

启动 HTTP 服务：

```bash
cd /path/to/ReFindVLA/Carla_code/Code
python3 model_server.py
```

服务默认监听 `0.0.0.0:8000`。另开终端检查：

```bash
curl http://127.0.0.1:8000/health
```

健康检查应返回模型名、设备、基座模型路径和 LoRA 路径。若控制端和 GPU 服务器不是同一台机器，应在控制端使用 GPU 服务器的局域网地址，例如：

```bash
export SERVER_URL=http://GPU_SERVER_IP:8000
```

不要把 Qwen 权重、服务器密码或访问令牌提交到 GitHub。

## 3. Linux 控制端：安装本地依赖

在运行 CARLA-Air/AirSim 的机器上：

```bash
cd /path/to/ReFindVLA/Carla_code
python3 -m venv .venv-client
source .venv-client/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-client.txt
```

如果 `carla` 没有安装到当前 Python 环境，设置 CARLA-Air 的安装目录。目录下需要有 `PythonAPI/`：

```bash
export CARLA_AIR_ROOT=/path/to/CarlaAir-v0.1.7
```

如果队员已经把 `carla` Python API 安装到环境中，可以不设置这个变量。AirSim Python API 也必须在当前环境中可导入。

## 4. 启动仿真器

先启动 CARLA-Air，再启动控制代码。当前本地 CARLA-Air README 中的 Linux 启动形式为：

```bash
cd /path/to/CarlaAir-v0.1.7
./CarlaAir.sh Town10HD
```

确认 CARLA 服务监听 `2000`，AirSim 服务监听 `41451`。本实验当前代码和已有数据记录以 Town10HD 为依据；不要在没有重新采集和评估的情况下把 Town3/Town5 写成已验证结果。

## 5. 启动在线 ReFindVLA 控制

在控制端设置服务地址和日志目录：

```bash
cd /path/to/ReFindVLA/Carla_code
source .venv-client/bin/activate
export CARLA_AIR_ROOT=/path/to/CarlaAir-v0.1.7
export SERVER_URL=http://GPU_SERVER_IP:8000
export TRACKING_LOG_DIR=/path/to/outputs/decision_runs
export RECOVER_USE_REMOTE_VLA=1
```

启动：

```bash
cd /path/to/ReFindVLA/Carla_code/Code
python3 -m main_interactive_track.main
```

程序会：

1. 检查远程 Qwen 服务；
2. 连接 CARLA-Air 和 AirSim；
3. 交互式读取目标指令；
4. 根据指令生成目标车和背景交通；
5. 在目标丢失后维护 Target Memory、Belief Map 和道路上下文；
6. 向 Qwen 请求高层 `phase/view/...` 决策；
7. 由本地航点规划器和安全控制器执行；
8. 将 `decision.jsonl`、控制帧和 sidecar 写入日志目录。

可使用的指令形式包括：

```text
找回失踪的白色厢式卡车，它最后向东行驶
spawn 白色 卡车
quit
```

## 6. 无 VLA 对照模式

当前代码提供一个不调用远程 VLA 的本地规则模式，用于保持其他模块不变时隔离 VLA 决策模块：

```bash
export RECOVER_USE_REMOTE_VLA=0
export RECOVER_REMOTE_REQUIRED=0
python3 -m main_interactive_track.main
```

此模式仍保留目标记忆、Belief Map、候选验证、航点规划和安全控制，只跳过 Qwen 远程请求。它是用于新增消融实验的代码开关，不代表已有论文结果已经包含该对照实验。

## 7. 阈值版本必须记录

当前清理后代码默认值为：

```text
confirm_threshold = 0.72
inspect_threshold = 0.52
memory_update_threshold = 0.70
candidate_margin = 0.06
```

现有实验资料中另有一组历史批次配置：

```text
confirm_threshold = 0.55
inspect_threshold = 0.45
memory_update_threshold = 0.55
```

两组参数不能混写。使用历史结果时，必须在实验记录中注明历史配置；使用当前代码进行新实验时，必须记录当前环境变量和 Git commit。

## 8. 决策级数据采集

数据采集不需要 Qwen 服务。从 `Code/` 目录执行：

```bash
cd /path/to/ReFindVLA/Carla_code/Code
python3 -m collect_data_code.record_find_track_data \
  --output-root /path/to/outputs/find_track_data \
  --episodes 1
```

采集器保存图像、目标框、状态和 `collection_profile.json`。它使用 CARLA 投影得到的仿真目标框，不是独立 YOLO 检测器，也不是独立 ReID 模型。

## 9. 运行前检查

提交或开始实验前，至少执行：

```bash
python3 -m compileall -q Code/main_interactive_track Code/collect_data_code Code/model_server.py
python3 -c "import torch, transformers, peft, fastapi, uvicorn; print('server imports ok')"
python3 -c "import numpy, requests, cv2; print('client imports ok')"
```

真正的 CARLA-Air、AirSim、Qwen GPU 和网络连通性只能在对应 Linux 机器上验证，不能用本地静态编译替代。

## 10. 不应提交或误读的内容

- 不提交完整 Qwen 基座权重；
- 不提交服务器密码、令牌、私有路径和个人数据；
- 不把旧版持续跟踪代码作为实验入口；
- 不把 `vx/vy/vz/yaw_rate` 写成 Qwen 的输出；
- 不把 CARLA 投影框写成真实视觉检测器或真实 ReID；
- 不把本手册中的命令执行成功写成实验结果；
- 新增无 VLA 对照实验之前，不在论文中报告其数值。
