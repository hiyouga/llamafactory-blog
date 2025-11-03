---
date: '2025-10-21T16:21:12+08:00'
draft: false
title: 'Megatron-Core Fine-Tuning with LLaMA-Factory'
author: 'LLaMA-Factory Team'
---

# LLaMA-Factory 🤝 MCoreAdapter

To fully leverage Megatron-core's parallel computing and improve training efficiency for MoE models, we combined the MCoreAdapter provided by the [ROLL team](https://github.com/alibaba/ROLL/tree/main/mcore_adapter) with LLaMA-Factory's data pipeline and Megatron Trainer's backend to build a new model training workflow.

## 🚀 Quick Start

### 1. 💻 Environment Installation

#### 📦 pip

```bash
# for megatron-core
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install \
    numpy==1.26.4 \
    optree>=0.13.0 \
    spacy==3.7.5 \
    weasel==0.4.1 \
    transformer-engine[pytorch]==2.2.0 \
    megatron-core==0.13.0 \
    deepspeed==0.16.4 

pip uninstall -y opencv opencv-python opencv-python-headless
pip install opencv-python-headless==4.11.0.86
pip install "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"

# for llamafactory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

#### 🐳 docker (Recommended)

Refer to the [Dockerfile](https://github.com/hiyouga/LLaMA-Factory/blob/main/docker/docker-cuda/Dockerfile.megatron) for building.

### 2. 🎯 Start Test

#### 🖥️ Single Node 8*80GB

```bash
cd LLaMA-Factory
# qwen2_vl_full
USE_MCA=1 llamafactory-cli train examples/megatron/qwen2_vl_full.yaml
# qwen3_moe_full
USE_MCA=1 llamafactory-cli train examples/megatron/qwen3_moe_full.yaml
```

#### 🌐 Multi Node 16*80GB

```bash
export DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
USE_MCA=1 torchrun $DISTRIBUTED_ARGS src/train.py \
    --model_name_or_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --do_train \
    --stage sft \
    --finetuning_type full \
    --dataset identity,alpaca_en_demo \
    --preprocessing_num_workers 16 \
    --cutoff_len 4096 \
    --template qwen3_nothink \
    --output_dir saves/mca/qwen3_moe_full_id \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 100 \
    --learning_rate 3e-6 \
    --logging_steps 1 \
    --save_steps 50 \
    --lr_scheduler_type constant \
    --bf16 \
    --tensor_model_parallel_size 1 \
    --sequence_parallel false \
    --pipeline_model_parallel_size 2 \
    --bias_activation_fusion true \
    --apply_rope_fusion true \
    --use_distributed_optimizer true \
    --overlap_param_gather true \
    --overlap_grad_reduce true \
    --moe_grouped_gemm true \
    --moe_token_dispatcher_type alltoall \
    --expert_model_parallel_size 4 \
    --recompute_granularity full
```

#### 📊 Benchmarks

We provide experiments for both multimodal and text MoE models. Refer to [this GitHub issue](https://github.com/hiyouga/LLaMA-Factory/pull/9237#issue-3492236945) for details.

#### 🔄 Weight conversion (mcore2hf)

You need to merge MCore type checkpoints saved during training into Hugging Face named safetensors using the conversion script:

```bash
python scripts/megatron_merge.py \
    --checkpoint_path saves/mca/qwen3_moe_full_id/checkpoint-50/ \
    --output_path saves/qwen3_moe_hf \
    --bf16
```

### 3. 💡 Tips & Precautions

#### 3.1 📐 Global Batch Size calculation differences

While using Megatron for training, note the subtle difference in how global batch size is calculated compared to previous setups:

** 📌 Parameter definitions: **

- `bs`: per_device_train_batch_size
- `ga`: gradient_accumulation_steps
- `ws`: WORLD_SIZE
- `pp`: pipeline_model_parallel_size
- `tp`: tensor_model_parallel_size
- `ep`: expert_model_parallel_size

**🔢 Formula comparison:**

```bash
# Original calculation
fsdp_global_batch_size = ws * bs * ga

# MCA calculation
mca_global_batch_size = (ws // pp // tp // ep) * bs * ga 
```

#### 3.2 ⚡ Performance optimization

- **💾 GPU memory optimization**: enable `--use_distributed_optimizer` and `--overlap_param_gather` would significantly reduce GPU memory usage
- **📡 Communication optimization**: use `--overlap_grad_reduce` to overlap gradient communication with computation
- **🔧 MoE optimization**: For MoE models, prefer `--moe_token_dispatcher_type alltoall` and `--moe_grouped_gemm true` for better performance
- **⚙️ Parallel optimization**: set `gradient_accumulation_steps` to be an integer multiple of PP

#### 3.3 🔍 Troubleshooting

1. **💥 OOM Errors**: reduce `per_device_train_batch_size` or `gradient_accumulation_steps`
2. **🌐 Communication timeouts**: check network connectivity, `master_addr` and `master_port`
3. **⚙️ Parallel settings**: ensure `pp * tp * ep` divides `ws` evenly
