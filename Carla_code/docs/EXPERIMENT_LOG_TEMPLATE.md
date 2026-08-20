# ReFindVLA 实验日志模板

每个策略批次和每个 episode 都必须填写或自动生成一份对应记录。没有填写的字段使用 `TBD`，不要凭记忆补写。

## 1. 批次元数据

```yaml
experiment_id: TBD
strategy: E1_full_remote_vla | E2_no_vla | E3_astar | E4_spiral
status: planned | smoke_test | completed | failed | excluded
date_utc: TBD
operator: TBD
repository: txh20040414/ReFindVLA
branch: experiment-design
commit: TBD
map: Town10HD
carla_air_version: v0.1.7
carla_api_version: 0.9.16
airsim_version: 1.8.1
episode_manifest: TBD
episode_count: TBD
matched_with: TBD
```

## 2. 计算与模型

```yaml
control_host: TBD
gpu_host: TBD
gpu_devices: TBD
gpu_memory_gb: TBD
driver: TBD
cuda: TBD
python: TBD
torch: TBD
transformers: TBD
peft: 0.10.0
base_model: /data/models/qwen/Qwen2.5-VL-7B-Instruct
lora_path: /data/ReFindVLA/Carla_code/models_/mode_v2
adapter_sha256: TBD
```

## 3. 运行配置

```yaml
server_url: TBD
server_health_checked: false
remote_vla: true
remote_required: true
control_dt_s: 0.5
decision_interval_s: 5.0
max_steps: 240
background_traffic: 10
confirm_threshold: 0.72
inspect_threshold: 0.52
memory_update_threshold: 0.70
candidate_margin: 0.06
image_width: 640
image_height: 480
```

E2 no-VLA 批次必须改为：

```yaml
remote_vla: false
remote_required: false
```

## 4. Episode 记录

| episode_id | seed | target type/color | loss trigger | policy | success | TTR (s) | path (m) | duration (s) | collision | false reacq | excluded reason |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

每个 episode 的原始文件：

```text
raw/<strategy>/<episode_id>/decision.jsonl
raw/<strategy>/<episode_id>/episode_meta.json
raw/<strategy>/<episode_id>/environment_meta.json
raw/<strategy>/<episode_id>/command.txt
raw/<strategy>/<episode_id>/hardware.txt
```

## 5. 批次汇总

```yaml
N_total: TBD
N_success: TBD
RSR: TBD  # N_success / N_total，必须同时报告分子和分母
TTR_mean_s: TBD
TTR_median_s: TBD
path_mean_m: TBD
duration_mean_s: TBD
collision_rate: TBD
false_reacquisition_rate: TBD
candidate_evaluations: TBD
remote_requests: TBD
remote_latency_mean_ms: TBD
remote_latency_p95_ms: TBD
fallback_count: TBD
```

## 6. 质量检查

- [ ] `health` 在批次开始前成功；
- [ ] 记录了代码 commit、GPU、模型和 adapter 路径；
- [ ] 策略之外的配置与匹配批次一致；
- [ ] E1/E2 使用同一 episode manifest；
- [ ] 每个 episode 都能找到原始 `decision.jsonl`；
- [ ] RSR 分母是 episode 数，不是决策请求数；
- [ ] 异常 episode 有排除原因；
- [ ] 没有把 smoke test 或历史批次并入新批次；
- [ ] 结果表可以由原始日志重新生成；
- [ ] 未完成的字段仍为 `TBD`。
