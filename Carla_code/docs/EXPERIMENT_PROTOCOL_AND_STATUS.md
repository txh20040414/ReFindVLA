# 实验协议与当前数据状态

本文档只区分“本地资料中已经存在的结果”和“需要队员重新运行的实验”，不把计划数字写成结果。

## 已有本地记录

本地 `论文创新与文档/第二版论文数据.md` 记录了以下批次：

- 外部公平对比：每个方法取 22 个 episode；A* Waypoint、Exponential Spiral 和 RecoverVLA 的 RSR 分别为 22.7%、31.8% 和 90.9%（20/22）；
- RecoverVLA 独立完整批次：25 个 episode，RSR 为 92.0%（23/25），平均路径长度 422.5 m，平均 episode 时长 169.1 s；
- 组件诊断批次：Rule Search 22 个 episode、Vision-Only VLA 24 个 episode、No Verification 45 个 episode、RecoverVLA Full 23 个 episode。

这些批次不是同一随机种子、同一 episode 数量或同一阈值版本的严格配对实验，正文中必须标为独立诊断批次，不能写成完整的全组合消融。

## 当前代码协议

当前清理后的代码默认值为：

```text
RECOVER_USE_REMOTE_VLA=1
RECOVER_CONFIRM_THRESHOLD=0.72
RECOVER_INSPECT_THRESHOLD=0.52
RECOVER_MEMORY_UPDATE_THRESHOLD=0.70
RECOVER_CANDIDATE_MARGIN=0.06
RECOVER_CAMERA_CONTROL=0
RECOVER_BACKGROUND_TRAFFIC=8
```

本地历史实验资料还记录过 `0.55/0.45/0.55` 阈值组合。历史表格不能直接标注为当前默认配置的结果；运行时必须保存环境变量和 commit。

## 审稿人要求的最小新增对照

使用同一套 CARLA-Air 场景、目标车、背景交通、随机种子、控制周期、日志定义和候选验证器，只将：

```text
RECOVER_USE_REMOTE_VLA=1
```

改为：

```text
RECOVER_USE_REMOTE_VLA=0
RECOVER_REMOTE_REQUIRED=0
```

这样可以保留 Target Memory、Belief Map、候选验证、语义航点和安全控制，只去掉远程 Qwen 决策。该对照代码已经提供，但当前仓库不预先填写其成功率、路径或时间结果。

## 每次实验必须记录

- Git commit；
- CARLA-Air 地图和版本；
- episode 数量、随机种子和目标/背景车辆配置；
- `RECOVER_*` 环境变量；
- Qwen 基座模型路径、LoRA adapter 路径和服务端 GPU；
- 服务端响应延迟；
- `decision.jsonl` 所在目录；
- RSR、路径长度、TTR、episode 时长、碰撞和错误确认的计算脚本或统计方式。

## 不应在当前仓库中声称的内容

- 尚未运行的 no-VLA 对照数值；
- Town3/Town5 的已验证性能；
- 真机 UAV 性能；
- 独立视觉检测器或真实 ReID 性能；
- 由当前候选权重自动学习或统计校准得到的参数。
