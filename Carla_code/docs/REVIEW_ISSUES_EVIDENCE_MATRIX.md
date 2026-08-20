# ReFindVLA 审稿问题—代码证据—实验状态矩阵

本文件把审稿意见映射到当前仓库的代码、论文需要的解释和本轮实验状态。`pending` 表示尚未产生新数据，不能写成已完成；`documented` 表示可以仅通过正文或可复现文档回应；`future` 表示当前范围内不新增实验。

## 1. 核心定位与 VLA 贡献

| 审稿疑问 | 当前代码证据 | 论文/Response 处理 | 状态 |
|---|---|---|---|
| ReFindVLA 是否输出无人机底层动作？ | `Code/model_server.py` 的 `/predict` 返回 `phase/view/target_found/confidence/...`；`decision.py` 生成语义航点，`controller.py` 才生成底层控制量 | 明确称为基于 VL 模型的高层规划器，不声称是端到端底层 VLA 控制器 | documented |
| VLA 的实际贡献能否隔离？ | `config.py` 提供 `RECOVER_USE_REMOTE_VLA`；`recover_vla.py` 在关闭时不调用远程服务 | 在相同 manifest 下新增 E1/E2 成对实验，比较 RSR、TTR、路径、耗时和碰撞 | pending |
| no-VLA 是否仍保留其他模块？ | `decision.py` 的 local fallback 仍经过 belief、candidate verifier、waypoint planner | 在实验手册中固定 E2 运行方式，禁止将其描述为删除整个系统 | documented/pending |

## 2. 可复现模型与候选验证

| 审稿疑问 | 本地/仓库证据 | 论文处理 | 状态 |
|---|---|---|---|
| LoRA 参数不完整 | `models_/mode_v2/adapter_config.json`：`r=8`、`alpha=16`、`dropout=0.05`、`q/k/v/o_proj`、`CAUSAL_LM` | 补全 adapter 配置、优化设置、prompt、结构化输出和模型选择流程；没有日志的优化器参数保持 TBD | documented/部分 pending |
| 损失函数未说明 | 训练记录和论文方法部分 | 写 masked autoregressive JSON cross-entropy；不把推理门控分数写成训练损失 | documented |
| 候选分数是否真实实现？ | `candidate_verification.py` 明确实现权重 `0.35/0.25/0.20/0.10/0.10`、阈值 `0.72`、间隔 `0.06` | 说明这些是工程先验，不是学习得到或验证集校准得到；`reid=0.50` 是占位中性分 | documented |
| 是否存在真实 ReID 网络？ | 当前代码没有独立 ReID 模型；颜色只是弱先验，`reid` 固定为 `0.50` | 不得声称使用了 learned ReID；真实 ReID 留作后续工作 | documented |
| P_obs/P_unv 是否在当前代码启用？ | 当前清理代码的 belief 分支只使用已实现的运动、方向和路网先验；未启用的因素不能画成运行模块 | 删除或标注未实现组件，避免架构图与代码不一致 | documented |

## 3. 数据、场景和统计

| 审稿疑问 | 当前证据 | 需要补充 | 状态 |
|---|---|---|---|
| 遮挡、道路分支、远距消失、相似车辆、视场受限数量 | 采集器和 CARLA 投影框可保存场景元数据，但当前材料没有经过 episode-level census 的完整分类表 | 由实际 episode manifest 统计数量和分布；没有统计就写 TBD | pending |
| 训练/验证/评测如何构建 | 本地已有训练样本和微调记录，但划分规则需绑定最终 manifest | 记录场景、目标外观、运动模式和 episode 的划分规则，避免同一 episode 泄漏 | pending |
| episode 与 run 的关系 | `logger.py` 为每次运行建立独立 `recover_YYYYMMDD_HHMMSS` 目录 | 明确一个 episode 对应一次独立 run；高层请求不是 run | documented |
| 90.9% 如何得到？ | 论文已有 `20/22` 的匹配批次表述 | 只在对应批次和分母仍被原始日志确认时使用；新实验单独计算 | documented/pending |
| 表格和图的数值不一致 | 论文数据来自不同批次的历史记录 | 表格增加 batch、N、配置和统计口径；不把 22/23/25 批次混合 | documented |

## 4. 基线与泛化

| 审稿疑问 | 处理原则 | 状态 |
|---|---|---|
| 缺少近年学习型基线 | 相关工作中定性讨论；如果输入输出和 CARLA-Air 任务不匹配、没有可复现实现，则解释不做数值复现；不能把不可复现方法填入表格 | future/documented |
| 多基座模型泛化 | 当前实验基座固定为 Qwen2.5-VL-7B-Instruct；跨基座验证不在本轮最小闭环中 | future |
| 真机实验 | 当前证据只支持 CARLA-Air/AirSim 仿真；讨论传感器噪声、定位漂移和扰动差异 | future/documented |
| 穷举所有组件组合 | 本轮只做严格 no-VLA 和已有诊断；完整组合消融成本过高 | future/documented |

## 5. 系统局限和训练风险

| 审稿疑问 | 代码/数据证据 | 论文处理 | 状态 |
|---|---|---|---|
| 10--11 s 推理延迟和 5 s 决策间隔 | `model_server.py` 返回 `timing_ms`；`recover_vla.py` 使用异步请求，控制循环继续运行 | 报告 VL 推理、图像处理、候选验证和网络开销；承认当前不是硬实时 | documented/待新日志核对 |
| 1160 样本、216 步微调过拟合 | 本地训练记录 | 讨论仿真布局、车辆外观和运动模式外推边界；不把训练损失下降当作泛化证明 | documented |
| 计算硬件不清楚 | 用户给出的 RTX 4090 24GB 与 RTX 6000 24GB；远程实例仍需 `nvidia-smi` 核对 | 只写入实际运行批次的硬件和显存信息 | pending |
| 仿真到真实差距 | 当前没有真机传感器和定位链 | 放入 limitation/future work，不声称已验证实飞 | documented/future |

## 6. 出版与仓库交付

以下内容不依赖新 CARLA 实验，但必须在投稿包中检查：

- Springer LNCS 模板、页码、页数和源文件；
- 摘要中的 `of ReFindVLA` 空格；
- iThenticate 相似度由作者在最终稿提交前实际检查；
- 版权表单和注册由作者完成；
- 本仓库不提交 Qwen 基座权重、服务器密码、访问令牌、私有日志和原始大规模数据；
- GitHub 中的 LoRA adapter 只在确认许可和仓库大小限制允许后保留。
