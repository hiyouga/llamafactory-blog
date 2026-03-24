---
date: '2026-03-24T10:00:00+08:00'
draft: false
title: '使用 MindSpore HyperParallel 在昇腾上进行 LlamaFactory FSDP2 训练'
author: 'LlamaFactory Team'
---

# LlamaFactory + MindSpore HyperParallel

我们将 MindSpore 社区的并行训练 [HyperParallel](https://gitcode.com/mindspore/hyper-parallel) 作为 FSDP2 后端集成到 LlamaFactory，支持**昇腾 NPU** 和 NVIDIA GPU，用户只需在 FSDP2 工作流上添加一行配置即可启用。

## 快速开始

### 1. 环境安装

#### pip

```bash
# 安装 HyperParallel
git clone https://gitcode.com/mindspore/hyper-parallel
cd hyper-parallel
pip install -e .

# 安装 LlamaFactory
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]" --no-build-isolation

# 安装 PyTorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# 可选：安装 torch-npu 以支持昇腾 NPU
pip install torch-npu==2.7.1
```

### 2. 配置

HyperParallel 训练需要两个配置文件：**Accelerate FSDP2 配置**和 **LlamaFactory 训练配置**。

#### 2.1 Accelerate FSDP2 配置

使用现有的 `examples/accelerate/fsdp2_config.yaml` 或自行创建：

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

#### 2.2 LlamaFactory 训练配置

创建包含 `use_hyper_parallel: true` 的训练 YAML：

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

### 3. 启动训练

```bash
cd LlamaFactory

# 方式一：在 YAML 配置中添加 use_hyper_parallel: true
accelerate launch \
    --config_file examples/accelerate/fsdp2_config.yaml \
    src/train.py examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml

# 方式二：在命令行追加 --use_hyper_parallel True，无需修改 YAML
accelerate launch \
    --config_file examples/accelerate/fsdp2_config.yaml \
    src/train.py examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml \
    --use_hyper_parallel True
```

### 4. 检查点与导出

HyperParallel 检查点以标准 HuggingFace 格式保存，无需额外的权重转换，可以直接使用 `from_pretrained()` 加载。

### 5. 说明

- HyperParallel 目前支持 `sft` 阶段且 `finetuning_type: full`
- Accelerate FSDP2 的配置（混合精度、显存优化等）照常使用，详见 [Accelerate FSDP 文档](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
