---
date: '2025-10-21T16:21:12+08:00'
draft: false
title: 'Megatron-Core Fine-Tuning with LLaMA-Factory'
author: 'LLaMA-Factory Team'
---

# LLaMA-Factory 🤝 MCoreAdapter

为充分利用 Megatron-core 的并行技术并提高 MoE 模型的训练效率，我们将 [**ROLL 团队**](https://github.com/alibaba/ROLL/tree/main/mcore_adapter) 提供的 MCoreAdapter 与 LLaMA-Factory 的数据链路及 Megatron Trainer 的训练后端相结合，构建了一个新的模型训练工作流。

## 🚀 快速开始

### 1. 💻 环境安装

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

#### 🐳 docker (推荐)

参考 [Dockerfile](https://github.com/hiyouga/LLaMA-Factory/blob/main/docker/docker-cuda/Dockerfile.megatron) 进行构建。

### 2. 🎯 启动实验

#### 🖥️ 单机 8*80GB

```bash
cd LLaMA-Factory
# qwen2_vl_full
USE_MCA=1 llamafactory-cli train examples/megatron/qwen2_vl_full.yaml
# qwen3_moe_full
USE_MCA=1 llamafactory-cli train examples/megatron/qwen3_moe_full.yaml
```

#### 🌐 多机 16*80GB

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

#### 📊 基准测试

我们为多模态模型与文本 MOE 模型各提供了一组实验，详情请见 [GitHub 评论](https://github.com/hiyouga/LLaMA-Factory/pull/9237#issue-3492236945)。

#### 🔄 权重转换(mcore2hf)

我们需要通过权重转换脚本将训练存储下来的 Mcore 类型的训练权重合并为 Hugging Face 命名类型的 Safetensors。

```bash
python scripts/megatron_merge.py \
    --checkpoint_path saves/mca/qwen3_moe_full_id/checkpoint-50/ \
    --output_path saves/qwen3_moe_hf \
    --bf16
```

### 3. ⚙️ Megatron 策略配置

理解 Megatron 的并行方式与优化参数对于高效训练至关重要。以下是关键配置选项的详细说明：  

#### 3.1 🔀 并行策略

- **`tensor_model_parallel_size` (TP)**：将单个权重矩阵在多块 GPU 上拆分。适用于无法放入单块 GPU 的超大模型。但会增加通信开销，因此适度使用（通常 2-8）。
  - *建议*：从 1 开始，仅在模型无法放入显存时增加  

- **`pipeline_model_parallel_size` (PP)**：将模型各层在 GPU 间按流水线方式分配。可减少单块 GPU 的显存占用，但可能产生流水线空泡。
  - *建议*：使用 2、4、8 等 2 的幂；设置 `gradient_accumulation_steps` 为 PP 的整数倍以减少空泡  

- **`expert_model_parallel_size` (EP)**：将 MoE（专家混合）模型的专家分布到不同 GPU 上。大型 MoE 模型必需。
  - *建议*：MoE 模型通常设为 2-8，视专家数量而定  

- **`context_parallel_size` (CP)**：在序列维度上分割计算，用于超长上下文（> 32k tokens）。
  - *建议*：用于超长序列，通常为 1、2 或 4  

- **`virtual_pipeline_model_parallel_size` (VPP)**：创建虚拟的流水线阶段，通过交错前向/反向传播减少空泡。
  - *建议*：在使用 PP 时设为 2-4，以提升效率  

- **`sequence_parallel`**：在 TP 组内分发序列级的计算（LayerNorm、Dropout）。当 TP > 1 时可减少显存占用。
  - *建议*：当 `tensor_model_parallel_size > 1` 时启用  

#### 3.2 💾 显存优化

- **`recompute_granularity`**：通过在反向传播时重新计算激活来节省显存（以计算换显存）。
  - `full`：重新计算整个 Transformer 层（最大显存节省）  
  - `selective`：仅重新计算注意力部分（平衡方案）  
  - *建议*：先用 `selective`，若仍显存不足再切换到 `full`  

- **`moe_layer_recompute`**：对 MoE 层进行检查点保存，以减少激活显存占用（适用于 MoE 模型）。
  - *建议*：大型 MoE 模型且显存紧张时启用  

#### 3.3 🚀 性能优化

- **`moe_token_dispatcher_type`**：决定 Token 如何路由到不同专家。
  - `alltoall`：大多数情况下性能更好（推荐）  
  - `allgather`：适用于特定网络拓扑  
  - *建议*：一般使用 `alltoall` 提高吞吐  

- **`moe_grouped_gemm`**：将专家的计算分组，以提升 GPU 利用率。
  - *建议*：MoE 模型始终设为 `true`  

- **`moe_shared_expert_overlap`**：将共享专家计算与通信重叠进行。
  - *建议*：启用以隐藏通信延迟（MoE 模型）  

- **`overlap_grad_reduce`**：在分布式优化器中将梯度归约与反向计算重叠。
  - *建议*：在使用 `use_distributed_optimizer: true` 时启用，以提升吞吐  

---

### 4. 💡 技巧与注意事项

#### 4.1 📐 全局批大小的计算差异

在使用 Megatron 训练时，全局批大小的计算方式与之前的设置略有不同：

**📌 参数定义：**

- `bs`: 每设备训练批大小（per_device_train_batch_size）  
- `ga`: 梯度累积步数（gradient_accumulation_steps）  
- `ws`: WORLD_SIZE（总 GPU 数）  
- `pp`: pipeline_model_parallel_size  
- `tp`: tensor_model_parallel_size  
- `ep`: expert_model_parallel_size  
- `cp`: context_parallel_size  

**🔢 公式对比：**

```bash
# 原始计算方式
fsdp_global_batch_size = ws * bs * ga

# MCA 计算方式
mca_global_batch_size = (ws // pp // tp // ep // cp) * bs * ga
```

**💡 差异说明：**  
关键在于，Megatron 的并行策略（PP、TP、EP、CP）会将总 GPU 划分，使得用于数据并行的有效 GPU 数减少。只有剩余的 GPU 参与数据并行，从而直接影响全局批大小。  

**📊 示例：**
```bash
# 设置：16 块 GPU, PP=2, TP=2, EP=2, CP=1, bs=4, ga=2
# 数据并行尺寸 = 16 // 2 // 2 // 2 // 1 = 2
# 全局批大小 = 2 * 4 * 2 = 16

# 如果增加 CP=2 用于长上下文：
# 数据并行尺寸 = 16 // 2 // 2 // 2 // 2 = 1
# 全局批大小 = 1 * 4 * 2 = 8
```

---

#### 4.2 ⚡ 性能优化

- **💾 GPU 显存优化**：启用 `--use_distributed_optimizer` 和 `--overlap_param_gather` 可显著降低显存占用  
- **📡 通信优化**：使用 `--overlap_grad_reduce` 将梯度通信与计算重叠  
- **🔧 MoE 优化**：MoE 模型优先选择 `--moe_token_dispatcher_type alltoall` 和 `--moe_grouped_gemm true`  
- **⚙️ 并行优化**：将 `gradient_accumulation_steps` 设为 PP 的整数倍  
- **📏 长上下文优化**：当训练超长序列（>32k tokens）时，启用 `context_parallel_size`（通常 2-4）分布序列计算，减少显存压力  

---

#### 4.3 🔍 故障排查

- **💥 OOM（显存不足）错误**：减少 `per_device_train_batch_size` 或 `gradient_accumulation_steps`，或在长序列时启用上下文并行，并检查是否已启用 `use_distributed_optimizer`  
- **🌐 通信超时**：检查网络连接、`master_addr` 和 `master_port`  
- **⚙️ 并行设置**：确保 `pp * tp * ep * cp` 能整除 `ws`  
- **📉 全局批大小过小**：如果由于过高的并行度（PP/TP/EP/CP）导致全局批大小过小，考虑增加 `gradient_accumulation_steps` 或适当降低并行度  
