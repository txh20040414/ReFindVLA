# ReFindVLA LoRA adapter

This directory contains the LoRA adapter used by the local ReFindVLA Qwen2.5-VL inference service. The Qwen2.5-VL-7B-Instruct base model is not included and must be installed separately on the Linux GPU machine.

## Recorded adapter configuration

- Base model: `Qwen2.5-VL-7B-Instruct`;
- PEFT: `0.10.0`;
- LoRA rank: `r=8`;
- LoRA alpha: `16`;
- LoRA dropout: `0.05`;
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- Task type: `CAUSAL_LM`;
- `use_rslora=false`.

The local training record reports 1,160 training samples, 117 validation samples, 216 steps and a best validation loss of approximately 0.1869. These metadata do not specify the optimizer or learning rate, so those values are not inferred here.

## Loading

Set the base model and adapter paths before starting `Code/model_server.py`:

```bash
export QWEN_BASE_MODEL=/data/models/qwen/Qwen2.5-VL-7B-Instruct
export REFINDVLA_LORA_PATH=/path/to/ReFindVLA/Carla_code/models_/mode_v2
```

The adapter is trained for structured high-level decision generation. The local controller is responsible for converting the returned phase and view into semantic waypoints and low-level AirSim control.
