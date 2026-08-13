---
date: '2026-08-13T20:03:59+08:00'
draft: false
title: 'MOSS-VL Meets LlamaFactory: LoRA and Full Fine-Tuning Out of the Box'
---

You can now fine-tune MOSS-VL directly with LlamaFactory.

MOSS-VL support has been merged into the main branch of LlamaFactory. Whether you want to validate an idea efficiently with LoRA or fully adapt the model to a specific domain, you can use the familiar LlamaFactory workflows for data processing, training, checkpoint resumption, and inference.

This tutorial first walks through a general multimodal MOSS-VL fine-tuning workflow using the WebUI, with a public basketball video dataset as the running example. After training, we examine how the model improves on a real basketball clip, then use Chinese chess notation to demonstrate its ability to learn an entirely new specialized task.

## MOSS-VL and LlamaFactory

[MOSS-VL](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0708) is a multimodal model designed for image and video understanding. With this integration, LlamaFactory can correctly process MOSS-VL visual inputs, cross-modal attention masks, and training labels, providing an end-to-end training and generation pipeline.

The following workflows have been validated end to end:

- **LoRA fine-tuning:** Training, saving, checkpoint resumption, adapter inference, and weight merging are supported.
- **Full fine-tuning:** All parameters can be unfrozen, with training, saving, checkpoint resumption, and inference fully supported.
- **Multimodal generation:** Fine-tuned models can generate normally through both Chat and Predict with Generate workflows.

The integration has passed community review and is now part of the main branch: [LlamaFactory PR #10708](https://github.com/hiyouga/LlamaFactory/pull/10708).

## Two Fine-Tuning Methods, One Workflow

LoRA is suitable for rapid experimentation and capability injection with lower memory requirements. Full fine-tuning is better suited to scenarios where you need to substantially change domain behavior, output formats, or specialized knowledge. Switching between the two only requires changing `finetuning_type`:

```yaml
model_name_or_path: OpenMOSS-Team/MOSS-VL-Instruct-0708
template: moss_vl
stage: sft
do_train: true

# Choose lora or full
finetuning_type: lora

dataset: your_multimodal_dataset
output_dir: saves/moss_vl/sft
```

You remain free to choose the model and task. LlamaFactory provides a unified training entry point, data pipeline, distributed training, checkpoint management, and subsequent inference workflow.

## Fine-Tuning MOSS-VL with the WebUI

You do not need to write a training script before getting started. The LlamaFactory WebUI handles model selection, data configuration, and training monitoring. The following workflow applies to multimodal tasks involving images or videos. We use a public basketball video dataset to make every step reproducible.

### 1. Install the Environment

We recommend an isolated Python 3.11 or 3.12 environment. First install the PyTorch build that matches your local CUDA environment, then clone the latest LlamaFactory and install the MOSS-VL-specific dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
uv pip install -e .
uv pip install -r requirements/moss-vl.txt

llamafactory-cli version
```

`requirements/moss-vl.txt` installs the Transformers, TorchCodec, and other dependencies required by the MOSS-VL integration. For an initial experiment, LoRA is usually the more accessible option. Full fine-tuning is better suited to multi-GPU environments with ample memory.

### 2. Prepare Multimodal Data with a Public Basketball Dataset

This tutorial uses the open-source [saveerjain/basketball-events](https://huggingface.co/datasets/saveerjain/basketball-events) dataset. It contains short clips from real basketball games with structured event annotations. The data can be used to learn shots, assists, blocks, rebounds, jersey colors and numbers, shot types, and shot outcomes.

```bash
mkdir -p projects/basketball_event_sft/data
hf download saveerjain/basketball-events \
  --repo-type dataset \
  --local-dir projects/basketball_event_sft/data
```

Register the training file in `projects/basketball_event_sft/data/dataset_info.json` so that the WebUI can discover the dataset:

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

The user message in each video sample uses the `<video>` placeholder for visual input, while the `videos` field points to the corresponding file. Before training, verify that every video path in the annotation file resolves correctly under the data directory.

### 3. Launch the WebUI

Run the following command from the LlamaFactory root directory:

```bash
llamafactory-cli webui
```

### 4. Configure and Start Training

On the Train page, select `MOSS-VL-Instruct-0708`. The model path is automatically set to `OpenMOSS-Team/MOSS-VL-Instruct-0708`, and the chat template should be `moss_vl`. Select SFT as the training stage, then choose LoRA or Full according to your available resources.

<!-- Upload: 01-model-configuration.png -->
![Configure the MOSS-VL model, fine-tuning method, and chat template](https://github.com/user-attachments/assets/8d2306cc-a0a3-435c-8a09-a91de673743c)

Set the data directory to `projects/basketball_event_sft/data`, select `mossvl_basketball_events_train`, and click Preview dataset to verify that the samples are loaded correctly.

<!-- Upload: 02-dataset-configuration.png -->
![Configure the training stage, data directory, and dataset](https://github.com/user-attachments/assets/ad99a8fa-5025-46f8-9261-6877a9b6995a)

Next, configure the learning rate, epochs, batch size, gradient accumulation, checkpoint interval, and other training parameters.

<!-- Upload: 03-training-parameters.png -->
![Configure the training parameters](https://github.com/user-attachments/assets/a6de9158-1b57-4e9e-a28c-1639ad1d1480)

You can also freeze the vision tower, multimodal projector, or language model independently, and set the pixel ranges for image and video inputs.

<!-- Upload: 04-module-freezing.png -->
![Configure multimodal module freezing and pixel ranges](https://github.com/user-attachments/assets/51f6cd43-8dce-47f1-9af5-f6d52f0e1ba6)

Click Preview command to inspect the final configuration. Once everything is correct, click Start to monitor the training logs, progress, and loss curve from the page.

<!-- Upload: 05-training-task.png -->
![Preview the training command and monitor task status](https://github.com/user-attachments/assets/cc3cbba2-9edf-4de6-a828-0e288b809fe6)

## Example 1: Understanding a Real Jump Shot

After training, we return to the basketball task used throughout the workflow. In real games, players move quickly, camera angles change constantly, and the key action that determines an event label may last only a few seconds. This makes the task a useful test of whether the model has learned fine-grained video understanding.

We train MOSS-VL on real game clips to identify players, shot actions, and outcomes. The following clip was not part of the training set. Player 4 in green dribbles to the area near the free-throw line, stops in front of a defender, jumps, and makes a two-point jump shot.

**Basketball demo video**

<!-- Upload: mossvl-basketball-demo-h264.mp4 -->
{{< rawhtml >}}
<video controls preload="metadata" style="width: 100%;">
  <source src="https://github.com/user-attachments/assets/08f5cf8b-a0bf-4cc9-a495-2abe06815d1b" type="video/mp4">
  Your browser does not support this video.
</video>
{{< /rawhtml >}}

**Prompt**

> Analyze this basketball clip. The two teams are wearing White and Green jerseys. Identify all shots, assists, blocks, and offensive rebounds. For each event, specify the player's jersey color and number; for shots, also provide the shot type and make/miss result. Return valid JSON with an "events" array.

### Base: Most of the Scene Is Already Understood

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

The base model understands most of the clip. It correctly identifies player 4 in green, recognizes a shot attempt, and determines that the shot was made. Its only mistake is classifying the mid-range jump shot as a layup.

### Full SFT: Recovering the Decisive Detail

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

After one round of full fine-tuning, the model retains all of its correct judgments while accurately identifying the action as a two-point jump shot. The difference between a broad understanding and a precise description comes down to this critical detail.

| Model | Player | Event | Shot type | Result |
| --- | --- | --- | --- | --- |
| Base | Green No. 4 | Shot | Layup | Make |
| Full SFT | Green No. 4 | Shot | **2-pt jump shot** | Make |

This example shows how fine-tuning can make the model perceive real-world video in greater detail and describe events more reliably. Next, we move to a completely different task to see whether the same workflow can teach the model a new specialized language.

## Example 2: Reconstructing Chinese Chess Notation from Video

For the second example, we created a synthetic SFT dataset of Chinese chess videos. Each video uses a standard board rendering, with different pause durations between moves. Unlike basketball event classification, this task requires temporal observation, piece recognition, move understanding, and translation of the entire sequence into standard Chinese chess notation.

The following demo video did not appear in the training set. It is approximately 31.8 seconds long, contains 14 half-moves, and has a resolution of 640 x 720.

**Chinese chess demo video**

<!-- Upload: mossvl-xiangqi-demo.mp4 -->
{{< rawhtml >}}
<video controls preload="metadata" style="width: 100%; max-height: 720px;">
  <source src="https://github.com/user-attachments/assets/44c42b64-49b1-4283-9252-6b8119ba7cab" type="video/mp4">
  Your browser does not support this video.
</video>
{{< /rawhtml >}}

**Prompt**

> Identify the Chinese chess moves in the video in order, use standard Chinese notation, and return strict JSON: `{"moves_zh":[...]}`.

**Reference answer**

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

### Base and Full SFT on the Same Video

We evaluate the original MOSS-VL-Instruct-0708 model and the fully fine-tuned checkpoint-600 using exactly the same video and prompt. To present a complete base-model response, we use one of 38 samples that stopped naturally without obvious repetition. The selection did not consider answer correctness. The fine-tuned result is the complete direct output from checkpoint-600.

#### Before Training: Base Model

The following base result uses `temperature=0.7`, `top_p=0.9`, and `seed=331`. It stops naturally after 211 tokens and produces valid JSON. However, each array element incorrectly combines one move from each side, and the notation does not match the video, so the model has not completed the target task.

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

#### After Training: Full SFT checkpoint-600

The fine-tuned model returns the requested JSON array. All 14 predicted moves exactly match the reference notation, and generation stops normally after the answer is complete.

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

| Model | Output structure | Notation match | Generation stop |
| --- | --- | --- | --- |
| Base, selected roll | Valid JSON, but each item combines two moves | 0 / 14 | Natural stop, 211 tokens |
| Full SFT checkpoint-600 | Target JSON array with one move per item | 14 / 14 | Natural stop, 76 tokens |

### More Than a Hand-Picked Example

We also run a fixed evaluation on 200 synthetic videos excluded from training. Overall recognition quality is measured with **LCS Recall**: the length of the longest move sequence shared by the prediction and reference in the correct order, divided by the reference sequence length. After full fine-tuning, LCS Recall increases from **7.1%** to **83.1%**.

| Metric | Base | Full SFT checkpoint-600 |
| --- | --- | --- |
| LCS Recall | 7.1% | **83.1%** |

Using the standard LlamaFactory fine-tuning pipeline, MOSS-VL can effectively learn new video tasks, specialized notation systems, and strict output formats.

## Start Fine-Tuning MOSS-VL

From basketball events and Chinese chess notation to industrial video and domain-specific visual question answering, developers can now prepare their own multimodal data in LlamaFactory, train MOSS-VL with LoRA or full fine-tuning, and reuse the same workflows for saving, resuming, and inference.

- Model: [OpenMOSS-Team/MOSS-VL-Instruct-0708](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0708)
- Integration PR: [hiyouga/LlamaFactory#10708](https://github.com/hiyouga/LlamaFactory/pull/10708)
- Training framework: [LlamaFactory](https://github.com/hiyouga/LlamaFactory)
