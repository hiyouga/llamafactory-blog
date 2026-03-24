---
date: '2026-03-24T10:00:00+08:00'
draft: false
title: 'MindSpore HyperParallel FSDP2 Training on Ascend with LlamaFactory'
author: 'LlamaFactory Team'
---

# LlamaFactory + MindSpore HyperParallel

We integrated [HyperParallel](https://gitcode.com/mindspore/hyper-parallel) from the MindSpore community as an FSDP2 backend into LlamaFactory, supporting **Ascend NPU** and NVIDIA GPU. Just one extra config line on the FSDP2 workflow to get started.

## Quick Start

### 1. Environment Installation

#### pip

```bash
# Install HyperParallel
git clone https://gitcode.com/mindspore/hyper-parallel
cd hyper-parallel
pip install -e .

# Install LlamaFactory
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]" --no-build-isolation

# Install PyTorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Optional: install torch-npu for Ascend NPU
pip install torch-npu==2.7.1
```

### 2. Configuration

HyperParallel training requires two config files: an **Accelerate FSDP2 config** and a **LlamaFactory training config**.

#### 2.1 Accelerate FSDP2 Config

Use the existing `examples/accelerate/fsdp2_config.yaml` or create your own:

```yaml
# examples/accelerate/fsdp2_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16  # or fp16
num_machines: 1  # the number of nodes
num_processes: 2  # the number of GPUs in all nodes
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

#### 2.2 LlamaFactory Training Config

Create a training YAML with `use_hyper_parallel: true`:

```yaml
# examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml

### model
model_name_or_path: Qwen/Qwen3-VL-30B-A3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true
use_v1_kernels: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
disable_gradient_checkpointing: false

### HyperParallel
use_hyper_parallel: true

### dataset
dataset: llava_1k_en, llava_1k_zh
template: qwen3_vl
cutoff_len: 1024
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Qwen3-VL-30B-A3B-Instruct/full/sft
logging_steps: 1
save_steps: 500
max_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
seed: 1234
```

### 3. Start Training

```bash
cd LlamaFactory

# Option 1: use a YAML config with use_hyper_parallel: true
accelerate launch \
    --config_file examples/accelerate/fsdp2_config.yaml \
    src/train.py examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml

# Option 2: append --use_hyper_parallel True to any existing FSDP2 training config
accelerate launch \
    --config_file examples/accelerate/fsdp2_config.yaml \
    src/train.py examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml \
    --use_hyper_parallel True
```

### 4. Checkpoint & Export

HyperParallel checkpoints are saved as standard HuggingFace format. No additional weight conversion is needed — you can directly load the saved checkpoint with `from_pretrained()`.

### 5. Notes

- HyperParallel currently supports the `sft` stage with `finetuning_type: full`
- Accelerate FSDP2 configuration (mixed precision, memory optimization, etc.) applies as usual — refer to the [Accelerate FSDP documentation](https://huggingface.co/docs/accelerate/usage_guides/fsdp) for details
