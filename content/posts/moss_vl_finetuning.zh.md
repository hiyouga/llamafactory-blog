---
date: '2026-08-13T20:03:59+08:00'
draft: false
title: 'MOSS-VL 接入 LlamaFactory：LoRA 与全参微调开箱即用'
---

现在，你可以直接使用 LlamaFactory 微调 MOSS-VL 了。

MOSS-VL 的适配代码已经合入 LlamaFactory 主分支。无论是希望低成本验证想法的 LoRA，还是面向特定领域进行充分训练的全参数微调，都可以复用 LlamaFactory 熟悉的数据、训练、断点续训与推理工作流。

本文先介绍如何通过 WebUI 完成一次通用的 MOSS-VL 多模态微调，并以公开篮球视频数据贯穿数据准备与训练配置。完成训练后，我们先查看模型在真实篮球片段上的变化，再用中国象棋记谱展示它学习全新专业任务的能力。

## MOSS-VL × LlamaFactory

[MOSS-VL](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0708) 是面向图像与视频理解的多模态模型。通过本次适配，LlamaFactory 已能够正确处理 MOSS-VL 的视觉输入、跨模态注意力掩码与训练标签，并打通完整的训练和生成链路。

目前已经完成端到端验证：

- **LoRA 微调：**训练、保存、断点续训、Adapter 推理与权重合并均可用。
- **全参数微调：**全部参数解冻，训练、保存、断点续训与推理均可用。
- **多模态生成：**训练后可通过 Chat 和 Predict with Generate 路径正常生成。

相关适配已经通过社区评审并合入主线：[LlamaFactory PR #10708](https://github.com/hiyouga/LlamaFactory/pull/10708)。

## 两种微调方式，一套工作流

LoRA 适合快速实验和较低显存成本的能力注入；全参数微调适合希望更充分改变模型领域行为、输出格式或专业知识的场景。二者只需要切换 `finetuning_type`：

```yaml
model_name_or_path: OpenMOSS-Team/MOSS-VL-Instruct-0708
template: moss_vl
stage: sft
do_train: true

# 选择 lora 或 full
finetuning_type: lora

dataset: your_multimodal_dataset
output_dir: saves/moss_vl/sft
```

模型和任务仍然由用户自由选择；LlamaFactory 负责统一训练入口、数据管线、分布式训练、Checkpoint 管理和后续推理。

## 用 WebUI 完成一次 MOSS-VL 多模态微调

不需要先写训练脚本，也可以在 LlamaFactory 的 WebUI 中完成模型选择、数据配置和训练监控。下面是一套适用于图像、视频等多模态任务的通用流程；为了便于复现，我们使用公开篮球视频数据演示每一步。

### 1. 安装环境

建议使用独立的 Python 3.11 或 3.12 环境，并先安装与本机 CUDA 匹配的 PyTorch。随后拉取最新版 LlamaFactory，并补充 MOSS-VL 的专用依赖：

```bash
uv venv --python 3.12
source .venv/bin/activate

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
uv pip install -e .
uv pip install -r requirements/moss-vl.txt

llamafactory-cli version
```

`requirements/moss-vl.txt` 会安装与 MOSS-VL 适配的 Transformers、TorchCodec 等依赖。初次尝试建议优先选择 LoRA；全参数微调更适合多卡高显存环境。

### 2. 准备多模态数据（以公开篮球数据为例）

本教程使用开源的 [saveerjain/basketball-events](https://huggingface.co/datasets/saveerjain/basketball-events)。数据包含真实比赛短视频及结构化事件标注，可用于学习投篮、助攻、盖帽、篮板，以及球衣颜色、号码、投篮类型和命中结果。

```bash
mkdir -p projects/basketball_event_sft/data
hf download saveerjain/basketball-events \
  --repo-type dataset \
  --local-dir projects/basketball_event_sft/data
```

在 `projects/basketball_event_sft/data/dataset_info.json` 中注册训练文件，WebUI 就能识别这个数据集：

```json
{
  "mossvl_basketball_events_train": {
    "file_name": "annotations/train_qwen.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

视频样本中的用户消息使用 `<video>` 标记视觉输入，并通过 `videos` 字段指向对应文件。开始训练前，请确认标注文件中的视频路径能在数据目录下正确找到。

### 3. 启动 WebUI

在 LlamaFactory 根目录运行：

```bash
llamafactory-cli webui
```

### 4. 配置并启动训练

在 Train 页面选择 `MOSS-VL-Instruct-0708`，模型路径会自动填入 `OpenMOSS-Team/MOSS-VL-Instruct-0708`，模板使用 `moss_vl`。训练阶段选择 SFT，微调方法可根据资源选择 LoRA 或 Full。

<!-- Upload: 01-model-configuration.png -->
![MOSS-VL 模型、微调方式与模板配置](https://github.com/user-attachments/assets/8d2306cc-a0a3-435c-8a09-a91de673743c)

将数据目录设为 `projects/basketball_event_sft/data`，选择 `mossvl_basketball_events_train`，再点击 Preview dataset 检查数据是否正确。

<!-- Upload: 02-dataset-configuration.png -->
![训练阶段、数据目录与数据集配置](https://github.com/user-attachments/assets/ad99a8fa-5025-46f8-9261-6877a9b6995a)

随后在参数配置页面设置学习率、训练轮数、Batch size、梯度累积和保存间隔等参数。

<!-- Upload: 03-training-parameters.png -->
![训练参数配置](https://github.com/user-attachments/assets/a6de9158-1b57-4e9e-a28c-1639ad1d1480)

也可以按模块冻结视觉编码器、多模态投影层或语言模型，并设置图像和视频的像素范围。

<!-- Upload: 04-module-freezing.png -->
![多模态模块冻结与像素范围配置](https://github.com/user-attachments/assets/51f6cd43-8dce-47f1-9af5-f6d52f0e1ba6)

点击 Preview command 检查最终配置；确认无误后点击 Start，即可在页面中查看训练日志、进度和损失曲线。

<!-- Upload: 05-training-task.png -->
![训练命令预览与任务状态](https://github.com/user-attachments/assets/cc3cbba2-9edf-4de6-a828-0e288b809fe6)

## 示例一：让 MOSS-VL 看懂一记真实跳投

训练完成后，我们先回到贯穿上文的篮球任务。真实赛场中球员快速移动、机位不断变化，决定一次事件标签的关键动作往往只有几秒，这正适合检验模型是否真正学会了更细粒度的视频理解。

我们用一批真实比赛片段，让 MOSS-VL 学习识别球员、投篮动作和比赛结果。下面这段视频没有参与训练：绿衣 4 号运球来到罚球线附近，在防守人面前急停起跳，完成一记两分跳投并命中。

**篮球 Demo 视频**

<!-- Upload: mossvl-basketball-demo-h264.mp4 -->
{{< rawhtml >}}
<video controls preload="metadata" style="width: 100%;">
  <source src="https://github.com/user-attachments/assets/08f5cf8b-a0bf-4cc9-a495-2abe06815d1b" type="video/mp4">
  当前浏览器无法播放该视频。
</video>
{{< /rawhtml >}}

**输入指令**

> Analyze this basketball clip. The two teams are wearing White and Green jerseys. Identify all shots, assists, blocks, and offensive rebounds. For each event, specify the player's jersey color and number; for shots, also provide the shot type and make/miss result. Return valid JSON with an "events" array.

### Base：已经看懂大半

```json
{
  "events": [
    {
      "jersey": "Green",
      "number": "4",
      "event": "shot",
      "shot_type": "layup",
      "result": "make"
    }
  ]
}
```

原始模型已经看懂了大部分内容：它正确找到了绿衣 4 号，知道这是一次投篮，也判断出了命中；唯一的失误，是把这记中距离跳投说成了上篮。

### Full SFT：补上决定性的细节

```json
{
  "events": [
    {
      "jersey": "Green",
      "number": "4",
      "event": "shot",
      "shot_type": "2-pt jump shot",
      "result": "make"
    }
  ]
}
```

经过一轮全参数微调，模型保留了原本正确的判断，同时把动作准确识别为“两分跳投”。从“看个大概”到“说清楚发生了什么”，差别就在这个关键细节。

| 模型 | 球员 | 事件 | 投篮动作 | 结果 |
| --- | --- | --- | --- | --- |
| Base | 绿衣 4 号 | 投篮 | 上篮 | 命中 |
| Full SFT | 绿衣 4 号 | 投篮 | **两分跳投** | 命中 |

从“看个大概”到准确说出动作类型，篮球示例展示了微调如何让模型对真实世界视频看得更细、表达得更稳。接下来，我们换一个完全不同的任务，看看同一套训练流程能否教会模型一门新的专业“语言”。

## 示例二：从象棋视频还原中文棋谱

作为第二个例子，我们合成了一套中国象棋视频 SFT 数据：画面使用标准棋盘渲染，并让每一步之间保持不同的停留时间。与篮球事件分类不同，模型需要同时完成时序观察、棋子识别、走法理解，并把整段过程翻译成标准中文棋谱。

下面这段 Demo 视频没有出现在训练集中。它长约 31.8 秒，包含 14 个半回合，分辨率为 640 × 720。

**象棋 Demo 视频**

<!-- Upload: mossvl-xiangqi-demo.mp4 -->
{{< rawhtml >}}
<video controls preload="metadata" style="width: 100%; max-height: 720px;">
  <source src="https://github.com/user-attachments/assets/44c42b64-49b1-4283-9252-6b8119ba7cab" type="video/mp4">
  当前浏览器无法播放该视频。
</video>
{{< /rawhtml >}}

**输入指令**

> 请按顺序识别视频中的中国象棋着法，使用标准中文记谱，并严格返回 JSON：`{"moves_zh":[...]}`。

**参考答案**

```json
{
  "moves_zh": [
    "卒３进１", "相七进五",
    "马８进９", "马二进三",
    "炮８平６", "车一进一",
    "车９平８", "炮二平一",
    "马２进３", "兵七进一",
    "卒３进１", "马八进六",
    "象３进５", "车九平七"
  ]
}
```

### 同一段视频，Base 与 Full SFT 的回答

我们使用完全相同的视频与输入指令测试原始 MOSS-VL-Instruct-0708 和全参数微调后的 checkpoint-600。为了让训练前结果可以完整展示，Base 回答取自 38 次同提示词采样中自然停止、无明显循环的一次；筛选时不参考答案正确性。训练后结果则直接使用 checkpoint-600 的完整生成。

#### 训练前：Base 模型

下面是推荐的 Base 展示结果：`temperature=0.7`、`top_p=0.9`、`seed=331`。它在 211 tokens 后自然停止，JSON 可以解析；但每个数组元素合并了红黑双方两着，且棋谱内容与视频不符，因此仍未完成目标任务。

```json
{
  "moves_zh": [
    "炮二平五 车9平8",
    "兵三进一 马8进7",
    "兵七进一 车1平2",
    "炮八平六 马2进3",
    "兵三进一 车2进5",
    "兵七进一 马3进2",
    "炮二平四 车2平4",
    "兵三进一 马2退3",
    "兵三进一 炮8平6",
    "车一平二 车4进5",
    "车二进四 马3退5",
    "车二平三 车4平6",
    "车三进二 马5进6",
    "车三退一 车6平4",
    "车三平四 马6退8"
  ]
}
```

#### 训练后：Full SFT checkpoint-600

微调后的模型按照要求返回 JSON 数组，识别出的 14 步与参考棋谱逐字一致，并在答案完成后正常停止。

```json
{
  "moves_zh": [
    "卒３进１", "相七进五",
    "马８进９", "马二进三",
    "炮８平６", "车一进一",
    "车９平８", "炮二平一",
    "马２进３", "兵七进一",
    "卒３进１", "马八进六",
    "象３进５", "车九平七"
  ]
}
```

| 模型 | 输出结构 | 棋谱匹配 | 生成终止 |
| --- | --- | --- | --- |
| Base（推荐 roll） | 合法 JSON，但每项合并两着 | 0 / 14 | 自然停止，211 tokens |
| Full SFT checkpoint-600 | 目标 JSON 数组，一项一着 | 14 / 14 | 自然停止，76 tokens |

### 不仅是一个漂亮样例

我们还在 200 条未参与训练的合成视频上进行了固定验证，并采用 **LCS Recall** 衡量整体识别效果：计算预测棋谱与参考棋谱中按顺序一致的最长着法序列，再除以参考棋谱长度。全参数微调后，LCS Recall 从 **7.1%** 提升至 **83.1%**。

| 指标 | Base | Full SFT checkpoint-600 |
| --- | --- | --- |
| LCS Recall | 7.1% | **83.1%** |

通过 LlamaFactory 的标准微调链路，MOSS-VL 能够有效学习新的视频任务、专业记谱体系和严格输出格式。

## 开始你的 MOSS-VL 微调

从篮球事件、象棋棋谱、工业视频到垂直领域的图像问答，开发者现在可以直接在 LlamaFactory 中准备自己的多模态数据，选择 LoRA 或全参数模式训练 MOSS-VL，并复用同一套保存、续训和推理流程。

- 模型：[OpenMOSS-Team/MOSS-VL-Instruct-0708](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0708)
- 适配 PR：[hiyouga/LlamaFactory#10708](https://github.com/hiyouga/LlamaFactory/pull/10708)
- 训练框架：[LlamaFactory](https://github.com/hiyouga/LlamaFactory)
