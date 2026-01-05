---
date: '2026-01-05T15:42:25+08:00'
draft: false
title: 'Qwen3-VL 模型相关问题'
---

这篇博客关注 Qwen3-VL 模型的几个小问题，并给出相应的问题原因和解决办法。

## 1 Qwen3-VL 模型训练推理速度慢

问题：一些帖子和 issues 提到，在 torch=2.9 并且使用 Conv3D 的情况下，Qwen3-VL 的训练推理速度相较于 torch=2.8 有大幅退化，参考 https://github.com/pytorch/pytorch/issues/166122。

### 1.1 检查 kernel 调用区别

首先分别在 torch=2.8 和 torch=2.9 两个版本下测试了 Conv3D 的 cuda 调用，测试代码如下：

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
```

执行如下命令，可以得到 cuda 内核调用信息

```bash
nsys profile --trace=cuda,nvtx --stats=true -o conv3d_profile python test_torch_Conv3D.py
```

- torch=2.8

![image-20260104163106073](https://github.com/user-attachments/assets/9b5f8544-e22b-4981-b73f-76c7ad6ae4c1)

- torch=2.9

![image-20260104162951855](https://github.com/user-attachments/assets/4d254346-07a1-42a3-ab00-ec3b5c04b438)

可以发现，torch=2.9 调用的 kernel 和 torch=2.8 调用的 kernel 不一样，并且 torch=2.9 调用的 vol2col 明显要耗时的多。

### 1.2 探究 torch 调用 cuda kernel 的逻辑

在 [aten/src/ATen/native/Convolution.cpp#L404](https://github.com/pytorch/pytorch/blob/c22a1b4277d9155f48b3666fb12a8e98d2d82d51/aten/src/ATen/native/Convolution.cpp#L404) 的 use_cudnn 函数会决定是否使用 cuDNN 实现的 Conv3D，cuDNN 实现的 Conv3D 是一个性能非常高的 kernel，但是在 cuDNN 9.8 - 9.14 版本之间会出现问题，torch=2.9 禁用了它，导致使用了更低效的 kernel。

### 1.3 LlamaFactory 解决方案

LlamaFactory 建议用户在使用 Conv3D 时规避 torch=2.9，为此 LlamaFactory 在模型加载时检测是否使用 Conv3D，并且以报错级别给出提示，见 [src/llamafactory/model/loader.py#L210](https://github.com/hiyouga/LlamaFactory/blob/68119e55224886ef21fea66606c5f6dc5d63bc2b/src/llamafactory/model/loader.py#L210) 。

## 2 对 Base 模型做 Zero RL 时 think 标签问题

问题：一些帖子反映在对 Qwen3 模型做 Zero RL，遇到过训练后的模型的输出很难 follow `<think>` `</think>` 格式的问题。

由于 `<think>` 和 `<\think>` 会被设置成额外的 special token，在 base 模型中这两个 token 没有被训练过，所以 embedding 也是初始化的，所以会出现不输出该 token 的情况。

解决方法：替换 `<think>` 和 `<\think>` 为其他单词，比如 `<thinking>` 和 `<\thinking>`  或者其他。