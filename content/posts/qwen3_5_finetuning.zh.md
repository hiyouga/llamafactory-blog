---
date: '2026-03-03T16:41:05+08:00'
draft: false
title: '使用 LlamaFactory 微调最新一代 Qwen3.5 模型辨别人形机器人型号'
---

2026年伊始，从美国拉斯维加斯消费电子展（CES）到中国春晚，中国自主研发的人形机器人频频“破圈”，多家中国企业的产品和应用不仅在海外业界引发热议，更是在全球社交媒体平台和国际媒体不断“刷屏”。具身智能，被视为人工智能发展的下一阶段，其核心在于实现智能“大脑”与物理“身体”的深度耦合，从而将数据、算法与算力直接转化为改造客观世界的行动能力。而人形机器人，因其与人类相似的外形和功能，被视为具身智能的高阶形态和最佳载体，有望成为继智能手机、新能源汽车之后的新一代超级终端。

[LlamaFactory](https://github.com/hiyouga/LlamaFactory) 是一款开源低代码大模型微调框架，集成了业界最广泛使用的微调技术，支持通过 Web UI 界面零代码微调大模型，目前已经成为开源社区内最受欢迎的微调框架之一，GitHub 星标将近 7 万。

通义千问团队开源了新一代多模态大模型 Qwen3.5，本教程将聚焦于如何利用开源的 Qwen3.5-9B 模型，借助 LlamaFactory 这一开源低代码大模型微调框架，针对“辨别人形机器人型号”这一具体任务进行微调。我们希望通过这一实践，展示轻量化大模型如何赋能具身智能应用，让机器人不仅“看得见”，更能“看得懂”，从而为这场正在席卷全球的智能革命，贡献一份来自开源社区的实践力量。

## 运行环境要求

- 建议 GPU 显存不低于 32 GB

## 1. 安装 LlamaFactory

拉取 LlamaFactory 到本地

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
```

安装 LlamaFactory 环境依赖

```bash
pip install -e .
```

运行如下命令，如果显示 LlamaFactory 的版本，则表示安装成功。

```bash
llamafactory-cli version
```

## 2. 准备数据集

[Manus](https://manus.im/app) 是一款专注于复杂任务执行的通用型 AI 智能体，能够从规划到执行自主完成端到端的任务。我们使用 Manus 自动化构建数据爬取的工作流，相比于传统的写爬虫脚本等方法非常高效。例如可以使用如下的 prompt 完成数据集的获取：

```
我想微调一个能识别所有2026年春晚出现人形机器人型号的 Qwen3.5 模型，我希望你给我生成一个自动化数据集爬取的工作流，爬取2026年马年春晚的机器人以及国内和国外常见的人型机器人的图片及其型号，制造商等等一些描述信息，然后最终给我一个格式如下的 json 数据集：
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>请识别并描述图片中的机器人及其特征，特别注意其型号、生产商及应用场景。"
      },
      {
        "role": "assistant",
        "content": "图片展示了由 宇树科技 (Unitree Robotics) 生产的 Unitree H1 机器人。搭载了先进的视觉和深度传感器系统，使其具备强大的环境感知能力，能够实时构建三维地图，实现自主导航和障碍物规避。"
      }
    ],
    "images": [
      "images/unitree_h1_5.png"
    ]
  },
  ...
]
我希望每个机器人图片对应一个问题和回答（一条数据），且user的问题和assistant的回答不单调，尽管机器人型号一样，我也希望问题和回答有多样性。

最终我希望得到的数据集至少有200张机器人图片，给我一个中文版的的json格式数据集和一个英文版的json格式数据集，还有数据集对应的图片文件。
```

本教程提供了一份多轮对话数据集，链接为：[mllm_robot.zip](https://drive.google.com/file/d/1Bapd-3B_x8OW9mFOKiHTjG2GchKUBdUg/view?usp=sharing)。数据集中的样本为单轮对话形式，含有 405 条样本，每条样本都由一条用户指令和一条模型回答组成，微调过程中模型不断学习样本中的回答风格，从而达到识别机器人的目的。数据样例如下所示：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>请识别并描述图片中的机器人及其特征，特别注意其型号、生产商及应用场景。"
      },
      {
        "role": "assistant",
        "content": "图片展示了由 宇树科技 生产的 Unitree G1 (EDU) 机器人。这款高性价比人形机器人以其卓越的运动性能和紧凑设计脱颖而出。它不仅支持灵活的折叠存放，极大节省空间，还能完成原地跳跃等复杂动作，展现了其先进的动力学控制能力。G1 (EDU) 旨在为科研和教育领域提供一个开放且功能强大的平台，助力机器人技术的发展和创新人才的培养。"
      }
    ],
    "images": [
      "images/unitree_g1_1.png"
    ]
  }
]
```

您可以下载该数据集放到 `LlamaFactory/data` 下面，并且修改 `dataset_info.json` 文件，添加如下内容：

```
  "mllm_robot": {
    "file_name": "mllm_robot.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "mllm_robot_en": {
    "file_name": "mllm_robot_en.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
```

这样 LlamaFactory 可以识别到新加入的数据集。

## 3. 模型微调

### 3.1 启动 Web UI

做好前序准备工作后，直接运行下述命令就可以启动 Web UI。

```bash
llamafactory-cli webui
```

点击返回的 URL 地址，进入 Web UI 页面。

### 3.2 配置参数

进入 WebUI 后，可以根据您的需求切换语言。首先配置模型，本教程选择 **Qwen3.5-9B** 模型，微调方法修改为 **lora**。

![image-20260303152428178](https://github.com/user-attachments/assets/f6eb4c49-7cdb-406c-8d1a-e6963d74a90f)

数据集使用 `mllm_robot` 和 `mllm_robot_en`，学习率使用 `1e-4`，Epochs 选择 5。

![image-20260303152509242](https://github.com/user-attachments/assets/884a102a-b507-464f-aaae-db44a1fbfbd8)

### 3.3 启动微调

将输出目录修改为 `train_qwen3_5_9B`，训练后的模型权重将会保存在此目录中。点击「预览命令」可展示所有已配置的参数，如果您想通过代码运行微调，可以复制这段命令，在命令行运行。

![image-20260303152540908](https://github.com/user-attachments/assets/d5c6cd90-4b7e-4978-8f5f-a4998844374b)

启动微调后需要等待一段时间，待模型下载完毕后可在界面观察到训练进度和损失曲线。在 5090 下模型微调大约需要 30 分钟，显示“训练完毕/Finished”代表微调成功。

![image-20260303160358571](https://github.com/user-attachments/assets/e36aa989-57f2-4886-84ae-d9d66c30d533)

## 4. 模型对话

### 4.1 对话微调模型

选择「Chat」栏，将**检查点路径**改为 `train_qwen3_5_9B`，点击「加载模型」即可在 Web UI 中和微调后的模型进行对话。

![image-20260303161002113](https://github.com/user-attachments/assets/6a75d0e1-3fea-4b64-9d86-e65caa5026ba)

随机上传一张图片，让模型识别图片中的机器人

![image-20260303161632573](https://github.com/user-attachments/assets/1ffad2d9-18cb-4db2-b8bd-6436bd75ffa1)

模型能够正确识别出图片中的机器人为魔法原子设计的 MagicBot Z1 (2026春晚定制版) 机器人，说明微调效果比较好。

### 4.2 对话原始模型

点击「卸载模型」，点击检查点路径输入框**取消勾选**检查点路径，再次点击「加载模型」，即可与微调前的原始模型聊天。

![image-20260303164535031](https://github.com/user-attachments/assets/4eca83d5-d1dd-490a-90c9-49751b365d75)

模型并没有识别出图片中的机器人，反而认为机器人是由人扮演的。说明模型微调是有效的。

## 5. 总结

本次教程介绍了如何使用 Manus 和 LlamaFactory 框架，基于 Lora 微调 Qwen3.5-9B 模型，使其能够识别机器人型号，同时通过人工测试验证了微调的效果。在后续实践中，可以使用实际业务数据集，对模型进行微调，得到能够解决实际业务场景问题的本地领域多模态大模型。