---
date: '2026-01-05T15:42:25+08:00'
draft: false
title: 'Issues Related to the Qwen3-VL Model'
---

This blog post focuses on several practical issues related to the Qwen3-VL model, along with an analysis of their root causes and corresponding solutions.

## 1. Slow Training and Inference Speed of Qwen3-VL

**Problem:**  
Some posts and GitHub issues report that when using **torch=2.9** together with **Conv3D**, the training and inference speed of Qwen3-VL degrades significantly compared to **torch=2.8**. See the related discussion at:  
https://github.com/pytorch/pytorch/issues/166122

### 1.1 Comparing CUDA Kernel Invocations

We first compared the CUDA kernel calls of Conv3D under **torch=2.8** and **torch=2.9**. The test code is shown below:

```python
import torch
import torch.nn as nn

class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x

net = Glm4vVisionPatchEmbed(
    patch_size=14,
    temporal_patch_size=2,
    in_channels=3,
    hidden_size=1536,
)

net = net.to('cuda').bfloat16()

x = torch.randn(8192, 14 * 14 * 3 * 2).to('cuda').bfloat16()
y = net(x)
print(y.shape)

with torch.cuda.nvtx.range("Glm4vVisionPatchEmbed"):
    y = net(x)

torch.cuda.synchronize()
````

The following command was used to collect CUDA kernel invocation information:

```bash
nsys profile --trace=cuda,nvtx --stats=true -o conv3d_profile python test_torch_Conv3D.py
```

- torch=2.8

![image-20260104163106073](https://github.com/user-attachments/assets/9b5f8544-e22b-4981-b73f-76c7ad6ae4c1)

- torch=2.9

![image-20260104162951855](https://github.com/user-attachments/assets/4d254346-07a1-42a3-ab00-ec3b5c04b438)

As shown above, **torch=2.9 invokes different CUDA kernels compared to torch=2.8**, and the `vol2col` kernel used in torch=2.9 is **significantly more time-consuming**.

### 1.2 How PyTorch Decides Which CUDA Kernel to Use

In the function `use_cudnn` located at
[aten/src/ATen/native/Convolution.cpp#L404](https://github.com/pytorch/pytorch/blob/c22a1b4277d9155f48b3666fb12a8e98d2d82d51/aten/src/ATen/native/Convolution.cpp#L404), PyTorch determines whether to use the cuDNN implementation for Conv3D.

The cuDNN-based Conv3D implementation is highly optimized and normally delivers excellent performance. However, **issues were discovered in cuDNN versions 9.8–9.14**, and as a result, **torch=2.9 disables this path**, falling back to a much less efficient kernel implementation.

### 1.3 LlamaFactory’s Solution

LlamaFactory recommends **avoiding torch=2.9 when using Conv3D**. To enforce this, LlamaFactory detects whether Conv3D is used during model loading and raises an **error-level warning** if torch=2.9 is detected.
See:
[src/llamafactory/model/loader.py#L210](https://github.com/hiyouga/LlamaFactory/blob/68119e55224886ef21fea66606c5f6dc5d63bc2b/src/llamafactory/model/loader.py#L210)

## 2. `<think>` Token Issues When Applying Zero RL to the Qwen3-VL-Instruct Model

**Problem:**
Some users have reported that after applying Zero RL to the Qwen3-VL-Instruct model, the trained model has difficulty following the `<think>` and `</think>` output format.

The root cause is that `<think>` and `</think>` are added as **additional special tokens**. In the **base model**, these tokens have never been seen during pretraining, so their embeddings are randomly initialized. As a result, the model may fail to reliably generate these tokens.

**Solution:**
Replace `<think>` and `</think>` with other words, such as `<thinking>` and `</thinking>`, or any other tokens that are already present in the vocabulary.
