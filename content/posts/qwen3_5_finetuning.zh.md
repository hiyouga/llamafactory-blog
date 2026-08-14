---
date: '2026-03-03T16:41:05+08:00'
draft: false
title: '使用 LlamaFactory 微调最新一代 Qwen3.5 模型辨别人形机器人型号'
---

2026年伊始，从美国拉斯维加斯消费电子展（CES）到中国春晚，中国自主研发的人形机器人频频“破圈”，多家中国企业的产品和应用不仅在海外业界引发热议，更是在全球社交媒体平台和国际媒体不断“刷屏”。具身智能，被视为人工智能发展的下一阶段，其核心在于实现智能“大脑”与物理“身体”的深度耦合，从而将数据、算法与算力直接转化为改造客观世界的行动能力。而人形机器人，因其与人类相似的外形和功能，被视为具身智能的高阶形态和最佳载体，有望成为继智能手机、新能源汽车之后的新一代超级终端。

[LlamaFactory](https://github.com/hiyouga/LlamaFactory)是一款开源低代码大模型微调框架，集成了业界最广泛使用的微调技术，支持通过 Web UI 界面零代码微调大模型，目前已经成为开源社区内最受欢迎的微调框架之一，GitHub 星标将近 7 万。

近期，通义千问团队开源了新一代多模态大模型 Qwen3.5，具备以下增强特性：

- **统一的视觉-语言基础**：在多模态 token 上进行早期融合训练，在推理、编码、智能体和视觉理解等基准测试中，跨代际达到与 Qwen3 相当的水平，并超越 Qwen3-VL 模型。
- **高效混合架构**：门控 Delta 网络结合稀疏混合专家（Mixture-of-Experts）机制，在极低延迟和成本开销下实现高吞吐推理。
- **可扩展的强化学习泛化能力**：在百万级智能体环境中进行强化学习训练，任务分布逐步复杂化，从而实现强大的现实世界适应能力。
- **全球语言覆盖**：支持扩展至 201 种语言和方言，实现包容性的全球部署，并具备细致入微的文化与区域理解能力。
- **下一代训练基础设施**：相比纯文本训练，多模态训练效率接近 100%，并采用异步强化学习框架，支持大规模智能体脚手架和环境编排。


本教程将聚焦于如何利用开源的 Qwen3.5 模型，借助 LlamaFactory 这一开源低代码大模型微调框架，针对“辨别人形机器人型号”这一具体任务进行微调。我们希望通过这一实践，展示轻量化大模型如何赋能具身智能应用，让机器人不仅“看得见”，更能“看得懂”，从而为这场正在席卷全球的智能革命，贡献一份来自开源社区的实践力量。


## 1. 本地微调 Qwen3.5-9B

运行环境要求

* 建议 GPU 显存不低于 32 GB


### 1.1 安装 LlamaFactory

拉取 LlamaFactory 到本地

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
```

安装 LlamaFactory 环境依赖

```bash
pip install -e .
```

**[可选]** 可以安装 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 获得训练推理加速效果。推荐使用源码安装方式，不推荐使用 PyPI 安装，可能会导致性能衰退，安装命令如下：

```
# uninstall both packages first to ensure a successful upgrade
pip uninstall fla-core flash-linear-attention -y && pip install -U git+https://github.com/fla-org/flash-linear-attention
```

运行如下命令，如果显示 LlamaFactory 的版本，则表示安装成功。

```bash
llamafactory-cli version
```

### 1.2 准备数据集

[Manus](https://manus.im/app)是一款专注于复杂任务执行的通用型 AI 智能体，能够从规划到执行自主完成端到端的任务。我们使用 Manus 自动化构建数据爬取的工作流，相比于传统的写爬虫脚本等方法非常高效。例如可以使用如下的 prompt 完成数据集的获取：

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

### 1.3 模型微调

#### 1.3.1 启动 Web UI

做好前序准备工作后，直接运行下述命令就可以启动 Web UI。

```bash
llamafactory-cli webui
```

点击返回的 URL 地址，进入 Web UI 页面。

#### 1.3.2 配置参数

进入 WebUI 后，可以根据您的需求切换语言。首先配置模型，本教程选择**Qwen3.5-9B**模型，微调方法修改为**lora**。

![](https://github.com/user-attachments/assets/f6eb4c49-7cdb-406c-8d1a-e6963d74a90f)

数据集使用 `mllm_robot` 和 `mllm_robot_en`，学习率使用 `1e-4`，Epochs 选择 5。

![](https://github.com/user-attachments/assets/884a102a-b507-464f-aaae-db44a1fbfbd8)

#### 1.3.3 启动微调

将输出目录修改为 `train_qwen3_5_9B`，训练后的模型权重将会保存在此目录中。点击「预览命令」可展示所有已配置的参数，如果您想通过代码运行微调，可以复制这段命令，在命令行运行。

![](https://github.com/user-attachments/assets/d5c6cd90-4b7e-4978-8f5f-a4998844374b)

启动微调后需要等待一段时间，待模型下载完毕后可在界面观察到训练进度和损失曲线。在 5090 下模型微调大约需要 30 分钟，显示“训练完毕/Finished”代表微调成功。

![](https://github.com/user-attachments/assets/e36aa989-57f2-4886-84ae-d9d66c30d533)
### 1.4. 模型对话

#### 1.4.1 对话微调模型

选择「Chat」栏，将检查点路径改为 `train_qwen3_5_9B`，点击「加载模型」即可在 Web UI 中和微调后的模型进行对话。

![](https://github.com/user-attachments/assets/6a75d0e1-3fea-4b64-9d86-e65caa5026ba)

随机上传一张图片，让模型识别图片中的机器人

![](https://github.com/user-attachments/assets/1ffad2d9-18cb-4db2-b8bd-6436bd75ffa1)

模型能够正确识别出图片中的机器人为魔法原子设计的 MagicBot Z1 (2026春晚定制版) 机器人，说明微调效果比较好。

#### 1.4.2 对话原始模型

点击「卸载模型」，点击检查点路径输入框**取消勾选**检查点路径，再次点击「加载模型」，即可与微调前的原始模型聊天。

![](https://github.com/user-attachments/assets/4eca83d5-d1dd-490a-90c9-49751b365d75)

模型并没有识别出图片中的机器人，反而认为机器人是由人扮演的。说明模型微调是有效的。

## 2. 在线微调 Qwen3.5-35B-A3B

### 2.1. 数据准备

本教程同样使用多轮对话数据集[mllm_robot.zip](https://drive.google.com/file/d/1Bapd-3B_x8OW9mFOKiHTjG2GchKUBdUg/view?usp=sharing)。

#### 2.1.1 数据格式转换

登录[LlamaFactory Online](https://www.llamafactory.com.cn/register?utm_source=LlamaFactory_qwen3.5)平台，在左侧边栏选择”实例空间“，选择资源，处理数据则选CPU即可。


![](https://github.com/user-attachments/assets/be8ba259-d74d-48d3-9120-11482c7663ac)

平台提供了vscode、jupyter两种常用的代码工具：


![](https://github.com/user-attachments/assets/ce291661-52a0-4e28-a9c6-3f4dcdacd6d1)

需要把数据处理成如下格式的图文对：


![](https://github.com/user-attachments/assets/ff861d38-f287-4820-8f5b-5703327e7c52)

#### 2.1.2 数据上传

平台提供了JupyterLab上传、SFTP上传下载两种方式，方式一快速、稳定适合大数量的上传，小数据量可选用方式二jupyter上传数据。

##### 方式1（大数据量时推荐）

SFTP (SSH File Transfer Protocol) 是一种安全的文件传输协议，通过加密的 SSH 连接传输文件。您可以通过“文件管理”的SFTP上传/下载功能，传输数据集、模型或您的其他文件到文件管理中。

**下载并安装Cyberduck**，点击\[ [这里](https://cyberduck.io/download/)\]进入网址，根据需求选择相应的系统版本(Windows/macOS)进行下载。


![](https://github.com/user-attachments/assets/614b95b1-131b-4e61-bdb3-276f41c2151e)


![](https://github.com/user-attachments/assets/00ffccce-ce72-410f-b319-74c53d2d9d59)


![](https://github.com/user-attachments/assets/349a532c-8cfc-4f68-85fc-d3d0cc9e241c)


##### 方式2:

提供了更灵活、更强大的功能。JupyterLab具有直观的图形化界面，支持并排编辑多个文档和多种文件类型（Notebook(.ipynb)、脚本(.py)、Markdown、CSV 等）。您可以通过JupyterLab传输模型、数据集或其他文件。


![](https://github.com/user-attachments/assets/6bc8d89b-72cd-4659-bb15-2aaacc871a48)



#### 2.1.3 数据集注册

 在 `/workspace/llamafactory/data/dataset_info.json` 配置文件中（如下图），配置如下内容，注册数据集 `alpaca_robot_en1.json` 和 `alpaca_robot_val_en.json`。

```
    "alpaca_robot_en": {
        "file_name": "/workspace/user-data/datasets/alpaca_robot_en.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        },
        "customized_status": 8,
        "total_tokens": "57242",
        "num_samples": "405",
        "avg_tokens": "141.34"
    }
```
 
```
"alpaca_robot": {
        "file_name": "/workspace/user-data/datasets/alpaca_robot.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        },
        "customized_status": 8,
        "total_tokens": "49205",
        "num_samples": "367",
        "avg_tokens": "134.07"
    }
```
 
在 `/workspace/llamafactory/data/dataset_info.json` 中追加数据集。


![](https://github.com/user-attachments/assets/f97256a2-2473-4ee1-b2ef-eb8dd436d7d7)


#### 2.1.4 数据集检测

a. 返回[LlamaFactory Online](https://www.llamafactory.com.cn/register?utm_source=LlamaFactory_qwen3.5)控制台，单击左侧导航栏的“文件管理”。

b. 单击目标数据集右侧“操作”列的"数据集检测"，检测数据集。如下图所示，若“数据集格式检测”结果显示“符合”，则表示数据集符合格式要求。

c.检测通过后可【训练数据-文件管理】选择该数据集进行微调、评估。


![](https://github.com/user-attachments/assets/14f71264-fb18-4175-88d3-a4e9d3bfa372)

为了方便您使用，`alpaca_robot` 数据集已在 **[Llamafactory online](https://www.llamafactory.com.cn/register?utm_source=LlamaFactory_qwen3.5)** 平台预置，可在模型微调、模型评估页面【训练数据-公共数据】选择使用该数据集。


![](https://github.com/user-attachments/assets/6bf8a0fa-edf1-42d9-862d-94bc59ffa9e1)

### 2.2. 模型训练

我们使用LlamaFactory Online平台，通过任务模式微调任务， 数据集使用 `alpaca_robot`  和 `alpaca_robot_en`， 微调/评估操作详情如下所示:


![](https://github.com/user-attachments/assets/dae84f0e-25a0-46a9-a3c9-f3378045fec0)

配置模型与数据集后，系统将根据所需资源及其相关参数，动态预估任务运行时长及微调费用。


![](https://github.com/user-attachments/assets/e9ff1ac6-2337-4429-a8cd-b295c2bbfe0c)

**通过任务中心查看任务状态。**  在左侧边栏选择”任务中心“，即可看到刚刚提交的任务。可以通过单击任务框，可查看任务的详细信息、超参数、训练追踪和日志。


![](https://github.com/user-attachments/assets/614644db-9bf0-431e-9d9f-31b3fe3a48ee)

进入SwanLab进行训练追踪，可查看微调参数、训练洗哦啊过、系统信息、日志、环境信息。


![](https://github.com/user-attachments/assets/a0d36a32-a866-4e8f-839c-5d6e2e1cead8)


![](https://github.com/user-attachments/assets/698a09cc-0881-466f-a5d5-63a79efe0083)

任务完成后，模型自动保存在"文件管理->模型->output"文件夹中。可在"任务中心->基本信息->模型成果"处查看保存路径。


![](https://github.com/user-attachments/assets/893acb23-1b3e-4c23-ada9-47fc4405d11b)

### 2.3 模型对话

#### 2.3.1 对话微调模型

随机上传一张图片，让模型识别图片中的机器人。


![](https://github.com/user-attachments/assets/728dd1bb-81fb-45b3-b2fb-1d510695ac7a)

模型能够正确识别出图片中的机器人图片展示了由宇树科技 (Unitree Robotics) 生产的 Unitree H1 机器人 ，证明了微调的有效性。

#### 2.3.2 对话原始模型

随机上传一张图片，让模型识别图片中的机器人。


![](https://github.com/user-attachments/assets/65fd69c1-1e31-4148-b88d-abe57d6d292d)

模型并没有正确识别出图片中的机器人信息， 验证了**模型微调**的有效性 。

## 3. 总结
本次教程介绍了如何使用 Manus 和 LlamaFactory 框架，通过本地环境或在线平台[LlamaFactory Online](https://www.llamafactory.com.cn/register?utm_source=LlamaFactory_qwen3.5) Lora微调 Qwen3.5系列模型，使其能够识别机器人型号，同时通过人工测试验证了微调的效果。在后续实践中，可以使用实际业务数据集，对模型进行微调，得到能够解决实际业务场景问题的本地领域多模态大模型。

