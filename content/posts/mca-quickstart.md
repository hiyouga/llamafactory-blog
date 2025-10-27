---
date: '2025-10-21T16:21:12+08:00'
draft: true
title: 'Megatron Full Finetune with LLaMaFactory'
author: 'LLaMaFactory Team'
---
# LLaMaFactory✖️Mcore Adapter
为了利用上megatron-core中的各项并行技术和GroupGEMM，我们通过结合[ROLL团队](https://github.com/alibaba/ROLL)提供的mcore_adapter，结合llamafactory的数据链路和megatron-trainer的训练后端，提供一个新的模型训练工作流。


## 🚀 快速开始

### 1. 💻 环境安装
> 📦 pip
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
> 🐳 docker(推荐)

参考[dockerfile](https://github.com/Kuangdd01/LLaMA-Factory-X/blob/1cef3e5f3d06146442c60bedbb88af529f174512/docker/docker-cuda/Dockerfile.megatron)进行构建

### 2. 🎯 启动实验
> 🖥️ 单机八卡(80gb)
```bash
cd LLaMA-Factory
# qwen2_vl_full
USE_MCA=1 llamafactory-cli train examples/megatron/qwen2_vl_full.yaml
# qwen3_moe_full
USE_MCA=1 llamafactory-cli train examples/megatron/qwen3_moe_full.yaml
```
> 🌐 多机实验
```bash
export DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
USE_MCA=1 torchrun $DISTRIBUTED_ARGS src/train.py \
    --model_name_or_path ../model/qwen3_30b_a3b \
    --do_train \
    --stage sft \
    --finetuning_type full \
    --dataset identity \
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

### 2.1 🔄 权重转换(mcore2hf)
```bash
python scripts/megatron_merge.py \
    --checkpoint_path saves/mca/qwen3_moe_full_id/checkpoint-50/ \
    --output_path saves/qwen3_moe_hf \
    --bf16
```

### 3. 💡 Tips & 注意事项

#### 3.1 📐 Global Batch Size 计算差异
在使用 Megatron 训练时，注意 global batch size 的计算相较于之前的设置有细微区别：

**📌 参数说明：**
- `ga`: gradient_accumulation_steps (梯度累积步数)
- `ws`: WORLD_SIZE (总进程数)
- `pp`: pipeline_model_parallel_size (流水线并行大小)
- `tp`: tensor_model_parallel_size (张量并行大小)
- `ep`: expert_model_parallel_size (专家并行大小)

**🔢 计算公式对比：**
```bash
# 原始计算方式
origin_global_batch_size = ws * batchsize_per_device * ga

# MCA 计算方式
mca_global_batch_size = (ws // pp // tp // ep) * batchsize_per_device * ga 
```

#### 3.2 ⚡ 性能优化建议
- **💾 内存优化**: 启用 `--use_distributed_optimizer` 和 `--overlap_param_gather` 可以显著减少内存使用
- **📡 通信优化**: 使用 `--overlap_grad_reduce` 可以重叠梯度通信和计算
- **🔧 MOE 优化**: 对于 MOE 模型，建议使用 `--moe_token_dispatcher_type alltoall` 和 `--moe_grouped_gemm true` 获得更好的性能
- **⚙️ 并行优化**: `gradient_accumulation_steps` 为 PP 的整数倍

#### 3.3 🔍 常见问题排查
1. **💥 OOM 错误**: 减少 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps`
2. **🌐 通信超时**: 检查网络连接和 `master_addr`、`master_port` 设置
3. **⚙️ 并行度设置**: 确保 `pp * tp * ep` 能整除 `ws`
