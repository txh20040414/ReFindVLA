# ReFindVLA 复现实验与审稿问题验证路线

本文件是当前 `Carla_code` 对应的实验执行协议。它只规定实验条件、运行顺序、日志要求和论文替换规则；没有运行得到的数据不得填入结果表，也不得写成已验证结论。

## 1. 实验目标与边界

本轮返修只优先解决一个最重要的可检验问题：

> 在路网先验、Target Memory、Belief Map、候选验证、语义航点规划器和安全控制器保持不变时，远程 Qwen 高层决策是否带来可测量收益？

因此主实验只比较以下四类策略：

| 编号 | 策略 | 代码状态 | 本轮作用 |
|---|---|---|---|
| E0 | 单集 smoke test | 已有入口 | 检查环境，不作为论文结果 |
| E1 | 完整 ReFindVLA | `RECOVER_USE_REMOTE_VLA=1` | 主系统 |
| E2 | 严格 no-VLA 对照 | `RECOVER_USE_REMOTE_VLA=0` | 隔离远程 VLA 决策贡献 |
| E3 | A* | 需使用已存在且可核对的基线实现 | 传统几何基线 |
| E4 | 螺旋搜索 | 需使用已存在且可核对的基线实现 | 传统搜索基线 |

当前仓库已经实现 E1 和 E2。经检查，当前 `Carla_code` 中没有可直接确认的独立 A* 或螺旋搜索实现；E3/E4 只有在队员提供实际脚本或实现后才能加入运行批次，不能用文档中的占位名称代替实验。

## 2. 固定系统拓扑

本项目采用“本地 CARLA-Air + 远程 Linux GPU Qwen”拓扑：

```text
本地 Windows/Linux 控制机
  CARLA-Air + CARLA 0.9.16 + AirSim 1.8.1
  Code/main_interactive_track
          │ HTTP POST /predict
          ▼
远程 Linux GPU 服务器
  Qwen2.5-VL-7B-Instruct + ReFindVLA LoRA
  Code/model_server.py:8000
```

Qwen 只返回高层结构化 JSON：`phase`、`view`、`target_found`、`confidence`、`hold_seconds` 和 `reason`。`vx/vy/vz/yaw_rate` 仍由本地 `decision.py`、`controller.py` 和 AirSim 接口生成，不能在实验记录中写成模型输出。

## 3. 统一实验协议

除策略开关外，E1--E4 必须使用同一批场景清单和同一参数。建议以现有论文的 Town10HD 匹配批次为第一阶段；如果新增场景，必须生成新的 episode manifest 并重新说明分布。

### 3.1 固定环境

- CARLA-Air：本地资料记录的 `v0.1.7`；
- CARLA Python API：`0.9.16`；
- AirSim Python API：`1.8.1`；
- 地图：`Town10HD`；未经重新运行，不得把 Town3/Town5 写成已验证结果；
- 默认背景交通数：`10`；
- 控制步长：`0.5 s`；
- 高层决策间隔：`5.0 s`；
- 当前代码默认最大步数：`240`；若为复现旧批次使用其他步数，必须在日志中单独标记；
- 当前候选门控：确认阈值 `0.72`、检查阈值 `0.52`、记忆更新阈值 `0.70`、候选间隔 `0.06`。

### 3.2 匹配场景原则

每个 episode 必须记录并在 E1--E4 之间复用：

```text
episode_id
random_seed（如果仿真器和脚本支持）
map_name
target_vehicle_type
target_color_instruction
target_spawn_point / target initial state
background traffic configuration
loss or occlusion trigger
episode timeout / max_steps
```

E1 和 E2 必须使用同一 `episode_id`，而不是分别随机生成两批场景。若 CARLA-Air 的重置无法保证完全相同，应保存完整的场景 manifest，并在论文中将其称为“matched scenario protocol”，不要声称为严格确定性复现。

### 3.3 样本量与统计规则

- 第一优先级：复用现有论文的匹配场景数量，形成 E1/E2 的成对比较；当前论文材料中已有 `22` 集匹配统计，新的运行结果必须重新核对，不能自动继承 `90.9%`；
- 如果时间允许，扩展到 `50` 个 episode，并将 `N=50` 明确写入所有表格；
- 不得把 `N=22`、`N=23`、`N=25` 或 `N=50` 的批次合并成一个百分比；
- `RSR = 成功找回的 episode 数 / 评测 episode 总数`；
- 每个 episode 是一次独立 run；多次控制循环和多次高层请求不应被当作多个 run；
- 报告百分比时必须同时写分子、分母，例如 `20/22 = 90.9%`；
- 均值指标同时报告 `N`，并注明是按 episode 汇总还是按决策事件汇总。

## 4. 运行顺序

### 阶段 A：环境和代码检查

在控制机执行：

```bash
python3 -m compileall -q Code/main_interactive_track Code/collect_data_code Code/model_server.py
python3 -c "import numpy, requests, cv2; print('client imports ok')"
```

在远程 GPU 机执行：

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python3 -c "import transformers, peft, fastapi, uvicorn; print('server imports ok')"
```

还必须保存：

```bash
nvidia-smi
git rev-parse HEAD
```

当前论文中的 RTX 4090 24GB 和 RTX 6000 24GB 只有在实际运行机 `nvidia-smi` 核对后才能作为本批次硬件记录。

### 阶段 B：启动远程 Qwen 服务

在远程 GPU 机：

```bash
cd /data/ReFindVLA/Carla_code
source .venv-server/bin/activate
export QWEN_BASE_MODEL=/data/models/qwen/Qwen2.5-VL-7B-Instruct
export REFINDVLA_LORA_PATH=/data/ReFindVLA/Carla_code/models_/mode_v2
```

环境变量名称必须保持为 ASCII 字符：`REFINDVLA_LORA_PATH`。

启动并检查：

```bash
cd /data/ReFindVLA/Carla_code/Code
python3 model_server.py
```

另开终端：

```bash
curl http://127.0.0.1:8000/health
```

控制机能够访问远程服务器时，再设置：

```bash
export SERVER_URL=http://GPU_SERVER_IP:8000
```

### 阶段 C：先做一个 E1 smoke test

只运行一个 episode，确认以下事实：

1. 本地能连接 CARLA-Air 和 AirSim；
2. `SERVER_URL/health` 可访问；
3. `/predict` 能返回结构化高层 JSON；
4. 日志中有 `decision.jsonl`、控制记录和必要的 sidecar；
5. 服务返回的是高层决策，不是底层速度；
6. 没有旧版持续跟踪链的入口、字段或评估文件被调用。

smoke test 通过后才进入成批实验。smoke test 的成功不能计入论文 RSR。

### 阶段 D：成对运行 E1/E2

完整系统 E1：

```bash
export RECOVER_USE_REMOTE_VLA=1
export RECOVER_REMOTE_REQUIRED=1
python3 -m main_interactive_track.main
```

严格 no-VLA E2：

```bash
export RECOVER_USE_REMOTE_VLA=0
export RECOVER_REMOTE_REQUIRED=0
python3 -m main_interactive_track.main
```

E2 仍然保留 Target Memory、Belief Map、候选验证、语义航点规划器和安全控制器；它只跳过远程 Qwen 请求，由 `decision.py` 的本地 fallback 生成高层计划。这才是本轮用于隔离 VLA 决策模块的对照，不是旧版持续跟踪实验。

### 阶段 E：E3/E4 基线

E3/E4 必须满足：

- 使用相同 episode manifest；
- 使用相同初始状态、地图、背景交通和终止条件；
- 使用相同控制器和碰撞判定；
- 只替换高层搜索策略；
- 保存每个 episode 的原始轨迹和最终指标；
- 在运行前把真实脚本路径、commit 和命令填入实验日志。

当前仓库没有足够证据证明 E3/E4 已经可由本项目入口直接运行，所以在基线代码加入前，论文中只能保留已有历史结果或写成待验证，不能把本文件当作基线实现证明。

## 5. 必须保存的输出

每个策略、每个 episode 至少保存：

```text
raw/
  decision.jsonl
  control.jsonl（如果该批次生成）
  episode_meta.json
  environment_meta.json
  frames/（若启用）
  run_config.txt
  hardware.txt
  command.txt
summary.csv
```

`decision.jsonl` 中的时间戳、策略来源、候选分数、置信度、phase、waypoint 和远程推理时间必须保留。任何人工修改后的汇总表必须能够回溯到这些原始文件。

## 6. 指标与分析顺序

首先计算 episode-level 指标：

1. Reacquisition Success Rate（RSR）；
2. Time to Reacquisition（TTR）；
3. 路径长度；
4. episode duration；
5. collision rate；
6. false reacquisition / false confirmation rate。

其次计算决策级诊断：

1. 候选评估次数；
2. 候选确认与拒绝数量；
3. 高层请求延迟及其分项；
4. fallback 次数；
5. memory update 次数。

不能用决策事件数代替 episode 数作为 RSR 分母。不能因为服务返回了 JSON 就把一次请求计为成功找回。

## 7. 论文替换规则

只有满足以下条件才替换论文中的 `TBD` 或旧批次数字：

- 原始日志完整；
- E1/E2 使用同一 manifest；
- 代码 commit、模型路径、LoRA adapter 和硬件已记录；
- 汇总脚本能够重新生成表格；
- 分子、分母和异常 episode 已人工复核；
- 结果没有把 smoke test、历史批次和新批次混合。

如果某项没有完成，正文写“not evaluated in the present protocol”或“left for future work”，不要填入推测值。
