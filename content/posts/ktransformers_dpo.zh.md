---
date: '2025-12-23T15:51:07+08:00'
draft: false
title: 'KTransformers 联合 LLaMA-Factory 进行 RL-DPO 训练'
---

本教程演示了如何使用 **LLaMA-Factory** 框架，通过**直接偏好优化（Direct Preference Optimization，DPO）** 对语言模型进行微调。DPO 是一种基于人类偏好来训练模型的方法，能够使模型输出更加对齐人类期望，更加以用户为中心。

## 1 环境配置

**软硬件要求：CPU 支持 AMX，系统的 glibc 版本大于等于 2.32，建议 GPU 显存大于等于 32G。**

### Step 1: 创建 KTransformers 的 conda 环境

```bash
conda create -n Kllama python=3.12 # choose from : [3.11, 3.12, 3.13]
conda activate Kllama
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-12.8.0 cuda-runtime
```

### Step 2: 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Step 3: 安装 KTransformers

Option 1: 从 https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.4 下载并安装与 **Torch** 和 **Python** 版本相匹配的 **KTransformers** wheel 包。

✨ CUDA 版本可以与 wheel 文件名中标注的版本不同。

```bash
pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.4/ktransformers-0.4.4+cu128torch29fancy-cp312-cp312-linux_x86_64.whl
```

❗❗❗ wheel 的 python, coda, torch 的版本号必须与当前环境一致

Option 2: 从源码安装 KTransformers

```bash
git clone --depth 1 https://github.com/kvcache-ai/ktransformers.git
cd ktransformers/kt-sft
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" # set according to your GPU

pip install -r "requirements-sft.txt"
KTRANSFORMERS_FORCE_BUILD=TRUE pip install -v . --no-build-isolation
```

### Step 4: 安装 Flash-attention wheel

从 https://github.com/Dao-AILab/flash-attention/releases 下载并安装与 **Torch** 和 **Python** 版本相匹配的 **Flash-Attention** wheel 包。

```bash
# abi=True/False can find from below
# import torch
# print(torch._C._GLIBCXX_USE_CXX11_ABI)

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

❗❗❗ wheel 的 python, coda, torch 的版本号必须与当前环境一致，还需要检查 abi 是 True 还是 False

### Step 5: (可选) 如果想使用 flash_infer

```bash
git clone https://github.com/kvcache-ai/custom_flashinfer.git
pip install custom_flashinfer/
```

## 2 DPO 训练

### 2.1 准备模型

本篇博客使用 DeepSeek-V2-Lite-Chat 模型作为演示，如果有需要可以替换为其他模型。

### 2.2 配置训练参数文件

（1）examples/train_lora/deepseek2_lora_dpo_kt.yaml

```yaml
### model
model_name_or_path: DeepSeek-V2-Lite-Chat
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: dpo_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Kllama_deepseekV2_DPO
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### ktransformers
use_kt: true # use KTransformers as LoRA sft backend
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml
cpu_infer: 64
chunk_size: 8192
```

（2）examples/inference/deepseek2_lora_dpo_kt.yaml

```yaml
model_name_or_path: DeepSeek-V2-Lite-Chat 
adapter_name_or_path: saves/Kllama_deepseekV2_DPO
template: deepseek
infer_backend: ktransformers  # choices: [huggingface, vllm, sglang, ktransformers]
trust_remote_code: true

use_kt: true # use KTransformers as LoRA sft backend to inference
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml
cpu_infer: 32
chunk_size: 8192
```

### 2.3 训练模型

```bash
# For LoRA SFT
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek2_lora_dpo_kt.yaml
```

训练结果如下：

![image-20251223152536433](https://github.com/user-attachments/assets/5d12c6db-2a86-46e8-8484-f768118b5814)

![image-20251223152548458](https://github.com/user-attachments/assets/8e551085-47f2-470e-966d-a1b64103b79b)

### 2.4 模型推理

```bash
# For Chat with model after LoRA SFT
llamafactory-cli chat examples/inference/deepseek2_lora_dpo_kt.yaml
```

![image-20251223153114783](https://github.com/user-attachments/assets/ec4d473c-451e-467b-8958-f103bb6e004b)

2.5 使用模型 API

```bash
# For API with model after LoRA SFT
llamafactory-cli api examples/inference/deepseek2_lora_dpo_kt.yaml
```

![image-20251223153402473](https://github.com/user-attachments/assets/33237c86-1f1a-4823-b7e2-736cb803001c)

## 报错示例

- 环境安装错误

![f525c0db5631bbd7e5c8cf0ec1580104](https://github.com/user-attachments/assets/42e31aa5-f92a-4f4e-96a2-cb7c80a218fa)

PyTorch , Python , FlashAttention, cuda 都必须保证一致

- KTransformers 只支持带有 AMX 功能的 CPU

![adf7dad879efc4cf4155008019f9c1e6](https://github.com/user-attachments/assets/2e281d94-f413-4ba5-b928-6d0cd19c9d2b)
