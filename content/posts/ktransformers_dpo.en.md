---
date: '2025-12-23T15:51:07+08:00'
draft: false
title: 'RL-DPO Training with KTransformers and LLaMA-Factory'
---

This tutorial demonstrates how to fine-tune a language model using the **LLaMA-Factory** framework with **Direct Preference Optimization (DPO)**. DPO is a training method based on human preferences, enabling model outputs to better align with human expectations and be more user-centric.

## 1 Environment Setup

**Software & hardware requirements: CPU must support AMX, the system glibc version must be ≥ 2.32, and a GPU with at least 32 GB of VRAM is recommended.**

### Step 1: Create a Conda Environment for KTransformers

```bash
conda create -n Kllama python=3.12 # choose from : [3.11, 3.12, 3.13]
conda activate Kllama
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-12.8.0 cuda-runtime
```

### Step 2: Install LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Step 3: Install KTransformers

**Option 1:** Download and install a **KTransformers** wheel that matches your **Torch** and **Python** versions from
[https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.4](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.4)

✨ The CUDA version can be different from the version indicated in the wheel filename.

```bash
pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.4/ktransformers-0.4.4+cu128torch29fancy-cp312-cp312-linux_x86_64.whl
```

❗❗❗ The Python, CUDA, and Torch versions of the wheel **must exactly match** the current environment.

**Option 2:** Install KTransformers from source

```bash
git clone --depth 1 https://github.com/kvcache-ai/ktransformers.git
cd ktransformers/kt-sft
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" # set according to your GPU

pip install -r "requirements-sft.txt"
KTRANSFORMERS_FORCE_BUILD=TRUE pip install -v . --no-build-isolation
```

### Step 4: Install Flash-Attention Wheel

Download and install a **Flash-Attention** wheel that matches your **Torch** and **Python** versions from
[https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)

```bash
# abi=True/False can find from below
# import torch
# print(torch._C._GLIBCXX_USE_CXX11_ABI)

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

❗❗❗ The Python, CUDA, and Torch versions must match the environment, and you must also verify whether the ABI is `True` or `False`.

### Step 5: (Optional) Enable `flash_infer`

```bash
git clone https://github.com/kvcache-ai/custom_flashinfer.git
pip install custom_flashinfer/
```

## 2 DPO Training

### 2.1 Prepare the Model

This blog uses the **DeepSeek-V2-Lite-Chat** model as an example. You may replace it with another model if needed.

### 2.2 Configure Training Parameter Files

(1) `examples/train_lora/deepseek2_lora_dpo_kt.yaml`

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
template: deepseek
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

(2) `examples/inference/deepseek2_lora_dpo_kt.yaml`

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

### 2.3 Train the Model

```bash
# For LoRA SFT
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek2_lora_dpo_kt.yaml
```

Training results:

![image-20251223152536433](https://github.com/user-attachments/assets/5d12c6db-2a86-46e8-8484-f768118b5814)

![image-20251223152548458](https://github.com/user-attachments/assets/8e551085-47f2-470e-966d-a1b64103b79b)

### 2.4 Model Inference

```bash
# For Chat with model after LoRA SFT
llamafactory-cli chat examples/inference/deepseek2_lora_dpo_kt.yaml
```

![image-20251223153114783](https://github.com/user-attachments/assets/ec4d473c-451e-467b-8958-f103bb6e004b)

### 2.5 Use the Model API

```bash
# For API with model after LoRA SFT
llamafactory-cli api examples/inference/deepseek2_lora_dpo_kt.yaml
```

![image-20251223153402473](https://github.com/user-attachments/assets/33237c86-1f1a-4823-b7e2-736cb803001c)

## Error Examples


### Environment Installation Errors

![f525c0db5631bbd7e5c8cf0ec1580104](https://github.com/user-attachments/assets/42e31aa5-f92a-4f4e-96a2-cb7c80a218fa)

**PyTorch, Python, FlashAttention, and CUDA must all be version-compatible.**
Before installing FlashAttention and KTransformers using wheel packages, check the installed Python and Torch versions with the following command:

```bash
pip list
```

This will display the versions of all installed packages. Locate the Torch version, for example:

```
torch                    2.9.1
```

Based on the Python version and CUDA runtime version installed in **Step 1** of the environment setup, you can determine the correct wheel packages for FlashAttention and KTransformers.

Then download the corresponding versions from:

* [https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.4](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.4)
* [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)

In this blog, the environment uses:

* **Python** = 3.12
* **Torch** = 2.9.1
* **CUDA** = 12.8
* **Architecture** = x86

Therefore, the correct wheels to install are:

* [ktransformers-0.4.4+cu128torch29fancy-cp312-cp312-linux_x86_64.whl](https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.4/ktransformers-0.4.4+cu128torch29fancy-cp312-cp312-linux_x86_64.whl)
* [flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl)

The suffixes in the wheel filenames indicate:

* `cu`: CUDA version
* `torch`: PyTorch version
* `cp`: Python version
* `cxx`: C++ standard
* `abi`: whether the C++ ABI is enabled

### KTransformers Only Supports CPUs with AMX

![adf7dad879efc4cf4155008019f9c1e6](https://github.com/user-attachments/assets/2e281d94-f413-4ba5-b928-6d0cd19c9d2b)

**AMX** refers to **Intel Advanced Matrix Extensions**, a set of **hardware-accelerated matrix computation instructions** introduced by Intel for **server and high-performance CPUs**. It is primarily designed for **AI, deep learning, and HPC workloads**.

You can check whether your CPU supports AMX with the following command:

```bash
lscpu | grep amx
```

If the output contains something like:

```bash
amx_tile amx_int8 amx_bf16
```

then your CPU supports AMX.

If no such output appears, the CPU does **not** support AMX, and you will need to switch to a different machine.