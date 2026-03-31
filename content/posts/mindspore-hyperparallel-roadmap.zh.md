---
date: '2026-03-30T15:42:54+08:00'
draft: false
title: 'LlamaFactory x MindSpore HyperParallel 社区协作路标'
---

> **昇思社区** · HyperParallel 超节点并行库
>
> 文档版本：v1.0 | 更新日期：2026-03-30

## 项目愿景

HyperParallel 是 **昇思社区** 新提出的超节点并行训练架构，致力于简化昇腾超节点编程，释放算力潜能。我们希望协同 LlamaFactory 生态提供易用、高性能的分布式训练解决方案。我们的目标是让每一位开发者都能在 Ascend NPU 和 NVIDIA GPU 上高效训练大模型，降低大模型训练的门槛和成本。

本路线图概述了 LlamaFactory 与 MindSpore HyperParallel 社区协作的发展方向，涵盖并行能力扩展、硬件优化、后端支持等多个维度。

## 路线图总览

```
2026 Q2                    2026 Q3                    2026 Q4
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│  Phase 1    │          │  Phase 2    │          │  Phase 3    │
│  能力扩展    │   ───►   │  硬件深化     │   ───►   │  后端多元    │
└─────────────┘          └─────────────┘          └─────────────┘
    │                          │                          │
    ├─ TP/EP/CP混合并行         ├─ 高维TP等优化              ├─ MindSpore后端扩展
    ├─ 更多模型泛化              ├─ HyperMPMD三层并行        ├─ 图算融合组件优化
    └─ 更大模型规模              └─ HyperOffload UD链卸载    └─ 更多训练阶段支持
```

## Phase 1: 并行能力扩展 (2026 Q2)

**目标**：扩展 TP（张量并行）/EP（专家并行）/CP（上下文并行）等多维混合并行能力，支持更大规模模型训练。

| 特性 | 描述 | 优先级 | 状态 |
|-----|------|-------|-----|
| **TP-EP 混合并行** | 支持 MoE 模型的 TP+EP 组合并行策略 | P0 | 验证中 |
| **CP 长序列支持** | 支持上下文并行，突破显存限制训练超长序列 | P0 | 验证中 |
| **3D 并行 (DP-TP-PP)** | 完整的三维混合并行支持，适配千亿级参数模型 | P1 | 验证中 |
| **昇腾亲和Offload策略** | 提供NPU亲和的多级智能显存卸载策略 | P2 | 开发中 |

**技术要点**：
- 统一的声明式并行策略配置接口
- 高效的通信原语和调度算法
- 昇腾亲和的并行和显存策略

## Phase 2: 昇腾硬件深度优化 (2026 Q3)

### 2.1 高维张量并行 (High-Dimensional TP)

**目标**：扩展高维 TP 等昇腾亲和并行特性，提升 Atlas A5/A3/A2 上训练的效率和泛化性。

| 特性 | 描述 | 硬件适配 | 预期收益 |
|-----|------|---------|---------|
| **2D-TP** | 双维张量并行，降低通信开销；TP 规模越大（≥8）收益越显著 | A5/A3 | 通信量减少 30%+（TP≥8 时更优） |
| **TP-PP 混合** | TP+流水线并行组合 | A5/A3/A2 | 显存优化 20%+ |

> **注**：高维 TP 的通信优化效果随 TP 并行度增大而愈加明显——当 TP≥8 时，传统 1D-TP 的 All-Reduce 通信量已成为显著瓶颈，2D-TP 通过将通信拆分到两个维度，通信量较 1D-TP 降低幅度可超过 40%。

### 2.2 MPMD 多核并行优化 (HyperMPMD)

**目标**：通过细粒度 MPMD（Multiple Program Multiple Data）并行，解决 MoE、多模态、强化学习等场景中的计算负载不均衡问题，充分利用昇腾超节点对等互联架构的协同能力。

HyperMPMD 在三个维度上提供 MPMD 能力：

**维度一：子模型内核级并发**

利用昇腾 NPU 片上 AICube/AIVector 多核异构特性，在单卡内实现计算与通信的细粒度流水编排，解决 MoE 架构的通信掩盖难题。

| 特性 | 描述 | 硬件适配 | 预期收益 |
|-----|------|---------|---------|
| **片内多核 MPMD** | AICube 负责矩阵运算，AIVector 负责通信前处理，两者并行流水 | A5/A3 | 通信掩盖率从 60% 提升至 **90%** |

**维度二：子模型间并发均衡（Inter-sub-model Concurrency Balancing）**

将模型中异构子模块（如多模态模型的文本/图像/音频编码器）解耦为独立并发子图任务，通过动态调度消除流水线气泡。

**维度三：跨模型并发调度（Cross-model Concurrent Scheduling）**

集成 MPMD 运行时的 Single Controller 模式，在超节点池化算力资源中实现模型级并发，适配强化学习的异步架构。

**预估收益**：
- 通信掩盖率从 **60% → 90%**
- 消除多模态/MoE 场景 **10-40%** 流水线气泡
- 整体训练性能提升约 **15%**，集群资源利用率提升 **15%+**

### 2.3 智能显存卸载 (HyperOffload)

**目标**：基于 Use-Definition（UD）链分析，将远端内存访问提升为计算图中的一等操作，实现确定性的全局显存规划与计算-通信重叠，充分释放超节点分层存储池的潜力。

**技术方案**：HyperOffload 基于编译器的 Use-Definition 链对张量的定义点（Definition）与使用点（Use）进行全局生命周期分析，精确识别每个张量的最佳卸载/预取时机。突破了以往只针对权重（Weights）卸载的局限，实现了对训练推理全流程中 KV Cache、中间激活值（Activations）及优化器状态的深度分层管理。通过 UD 链驱动的统一逻辑视图，根据硬件拓扑自动感知 HBM 和 DDR 的带宽差异，将海量张量跨介质无缝调度。

**预估收益**（基于 HyperOffload 论文实验数据）：

*训练场景*：
| 模型 | 硬件配置 | 基线 | + HyperOffload | 性能变化 |
|-----|---------|------|----------------|---------|
| LLaMA-8B | 8×Ascend 910C | 5.2s/step | 4.08s/step | **提升 ~20%** |
| DeepSeek-V3 | 8×Ascend 910C | 2.5s/step | 2.19s/step | **提升 ~12%** |

## Phase 3: MindSpore 后端支持 (2026 Q4)

**目标**：LlamaFactory 官方支持 MindSpore 后端，使能 AKG、DVM 等 MindSpore 独有的深度图算融合优化能力，进一步释放昇腾 NPU 算力。

## 社区协作计划

### 与 LlamaFactory 社区的协作

| 协作领域 | 具体内容 | 负责方 |
|---------|---------|-------|
| **代码集成** | LlamaFactory 官方支持 MindSpore 后端，集成 HyperParallel 并行能力 | 共建 |
| **文档共建** | 在 LlamaFactory 官方文档中增加 MindSpore 后端使用指南 | 共建 |
| **Issue 处理** | 建立联合 Issue 处理机制 | 共建 |
| **版本同步** | 确保 HyperParallel 与 LlamaFactory 版本兼容性 | 共建 |

## 联系我们

- **MindSpore HyperParallel 仓库**: https://atomgit.com/mindspore/hyper-parallel
- **MindSpore 官网**: https://www.mindspore.cn/
- **MindSpore 社区**: https://gitcode.com/mindspore

## 附录：术语表

| 术语 | 全称 | 描述 |
|-----|-----|-----|
| TP | Tensor Parallelism | 张量并行 |
| EP | Expert Parallelism | 专家并行 |
| CP | Context Parallelism | 长序列并行 |
| DP | Data Parallelism | 数据并行 |
| PP | Pipeline Parallelism | 流水线并行 |
| FSDP | Fully Sharded Data Parallel | 全分片数据并行 |
| SPMD | Single Program Multiple Data | 单程序多数据并行 |
| MPMD | Multiple Program Multiple Data | 多程序多数据并行 |
| HCCL | Huawei Collective Communication Library | 华为集合通信库 |
