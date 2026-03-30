---
date: '2026-03-30T15:42:54+08:00'
draft: false
title: 'LlamaFactory x MindSpore HyperParallel Community Collaboration Roadmap'
---

> **MindSpore Community** · HyperParallel SuperNode Parallel Library
>
> Version: v1.0 | Updated: 2026-03-30

## Vision

HyperParallel is a new supernode parallel training architecture proposed by the **MindSpore Community**, dedicated to simplifying Ascend supernode programming and unlocking computing potential. We aim to collaborate with the LlamaFactory ecosystem to provide an easy-to-use, high-performance distributed training solution. Our goal is to enable every developer to efficiently train large models on Ascend NPU and NVIDIA GPU, lowering the barrier and cost of large model training.

This roadmap outlines the development direction of the LlamaFactory and MindSpore HyperParallel community collaboration, covering parallel capability expansion, hardware optimization, backend support, and more.

## Roadmap Overview

```
2026 Q2                    2026 Q3                    2026 Q4
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│  Phase 1    │          │  Phase 2    │          │  Phase 3    │
│  Capability │   ───►   │  Hardware   │   ───►   │  Backend    │
│  Expansion  │          │  Deepening  │          │  Diversity  │
└─────────────┘          └─────────────┘          └─────────────┘
    │                          │                          │
    ├─ TP/EP/CP Hybrid         ├─ High-Dim TP             ├─ MindSpore Backend
    ├─ More Model Coverage     ├─ HyperMPMD 3-Level       ├─ Graph-Kernel Fusion
    └─ Larger Model Scale      └─ HyperOffload UD-Chain   └─ More Training Stages
```

## Phase 1: Parallel Capability Expansion (2026 Q2)

**Goal**: Extend multi-dimensional hybrid parallel capabilities including TP (Tensor Parallelism), EP (Expert Parallelism), and CP (Context Parallelism) to support larger-scale model training.

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| **TP-EP Hybrid** | Support TP+EP combined parallelism for MoE models | P0 | Validating |
| **CP Long Sequence** | Context parallelism to break memory limits for ultra-long sequences | P0 | Validating |
| **3D Parallel (DP-TP-PP)** | Full 3D hybrid parallelism for 100B+ parameter models | P1 | Validating |
| **Ascend-Affinity Offload** | NPU-affinity multi-level intelligent memory offload strategies | P2 | In Development |

**Key Technical Points**:
- Unified declarative parallel strategy configuration interface
- Efficient communication primitives and scheduling algorithms
- Ascend-affinity parallel and memory strategies

## Phase 2: Ascend Hardware Deep Optimization (2026 Q3)

### 2.1 High-Dimensional Tensor Parallelism (High-Dimensional TP)

**Goal**: Extend high-dimensional TP and other Ascend-affinity parallel features to improve training efficiency and generalization on Atlas A5/A3/A2.

| Feature | Description | Hardware | Expected Benefit |
|---------|-------------|----------|-----------------|
| **2D-TP** | Two-dimensional tensor parallelism to reduce communication overhead; benefits grow significantly when TP ≥ 8 | A5/A3 | Communication reduced by 30%+ (even better at TP ≥ 8) |
| **TP-PP Hybrid** | TP + Pipeline Parallelism combination | A5/A3/A2 | Memory optimization 20%+ |

> **Note**: The communication optimization of high-dimensional TP becomes increasingly significant as TP parallelism degree grows—when TP ≥ 8, the All-Reduce communication volume of traditional 1D-TP becomes a major bottleneck. 2D-TP splits communication across two dimensions, reducing communication volume by over 40% compared to 1D-TP.

### 2.2 MPMD Multi-Core Parallel Optimization (HyperMPMD)

**Goal**: Leverage fine-grained MPMD (Multiple Program Multiple Data) parallelism to address computational load imbalance in MoE, multimodal, and reinforcement learning scenarios, fully utilizing the peer-to-peer interconnect architecture of Ascend supernodes.

HyperMPMD provides MPMD capabilities across three dimensions:

**Dimension 1: Intra-sub-model Core-Level Concurrency**

Leveraging the heterogeneous multi-core AICube/AIVector architecture on Ascend NPUs to achieve fine-grained compute-communication pipelining within a single card, addressing the communication masking challenge in MoE architectures.

| Feature | Description | Hardware | Expected Benefit |
|---------|-------------|----------|-----------------|
| **On-chip Multi-core MPMD** | AICube handles matrix ops, AIVector handles communication preprocessing, both pipelined in parallel | A5/A3 | Communication masking ratio from 60% to **90%** |

**Dimension 2: Inter-sub-model Concurrency Balancing**

Decoupling heterogeneous sub-modules (e.g., text/image/audio encoders in multimodal models) into independent concurrent subgraph tasks, eliminating pipeline bubbles through dynamic scheduling.

**Dimension 3: Cross-model Concurrent Scheduling**

Integrating the MPMD runtime's Single Controller mode to enable model-level concurrency within the supernode's pooled computing resources, supporting the asynchronous architecture of reinforcement learning.

**Expected Benefits**:
- Communication masking ratio from **60% → 90%**
- Eliminate **10-40%** pipeline bubbles in multimodal/MoE scenarios
- Overall training performance improvement of approximately **15%**, cluster resource utilization improvement of **15%+**

### 2.3 Intelligent Memory Offloading (HyperOffload)

**Goal**: Based on Use-Definition (UD) chain analysis, elevate remote memory access to a first-class operation in the computation graph, achieving deterministic global memory planning and compute-communication overlap, fully releasing the potential of the supernode's hierarchical storage pool.

**Technical Approach**: HyperOffload performs global lifetime analysis of tensor definition points and use points through the compiler's Use-Definition chain, precisely identifying the optimal offload/prefetch timing for each tensor. It goes beyond traditional weight-only offloading to enable deep hierarchical management of KV Cache, intermediate activations, and optimizer states throughout the training and inference pipeline. Through a UD chain-driven unified logical view, it automatically detects bandwidth differences between HBM and DDR based on hardware topology, seamlessly scheduling massive tensors across storage tiers.

**Expected Benefits** (based on HyperOffload paper experimental data):

*Training Scenarios*:
| Model | Hardware Config | Baseline | + HyperOffload | Performance Change |
|-------|----------------|----------|----------------|--------------------|
| LLaMA-8B | 8×Ascend 910C | 5.2s/step | 4.08s/step | **~20% improvement** |
| DeepSeek-V3 | 8×Ascend 910C | 2.5s/step | 2.19s/step | **~12% improvement** |

## Phase 3: MindSpore Backend Support (2026 Q4)

**Goal**: LlamaFactory officially supports MindSpore backend, enabling AKG, DVM and other MindSpore-exclusive deep graph-kernel fusion optimization capabilities, further unlocking Ascend NPU computing power.

## Community Collaboration Plan

### Collaboration with LlamaFactory Community

| Area | Details | Responsible |
|------|---------|-------------|
| **Code Integration** | LlamaFactory officially supports MindSpore backend, integrating HyperParallel parallel capabilities | Co-built |
| **Documentation** | Add MindSpore backend user guide to LlamaFactory official documentation | Co-built |
| **Issue Handling** | Establish joint issue handling mechanism | Co-built |
| **Version Sync** | Ensure HyperParallel and LlamaFactory version compatibility | Co-built |

## Contact Us

- **MindSpore HyperParallel Repository**: https://atomgit.com/mindspore/hyper-parallel
- **MindSpore Official Website**: https://www.mindspore.cn/
- **MindSpore Community**: https://gitcode.com/mindspore

## Appendix: Glossary

| Term | Full Name | Description |
|------|-----------|-------------|
| TP | Tensor Parallelism | Tensor Parallelism |
| EP | Expert Parallelism | Expert Parallelism |
| CP | Context Parallelism | Long Sequence Parallelism |
| DP | Data Parallelism | Data Parallelism |
| PP | Pipeline Parallelism | Pipeline Parallelism |
| FSDP | Fully Sharded Data Parallel | Fully Sharded Data Parallelism |
| SPMD | Single Program Multiple Data | Single Program Multiple Data |
| MPMD | Multiple Program Multiple Data | Multiple Program Multiple Data |
| HCCL | Huawei Collective Communication Library | Huawei Collective Communication Library |
