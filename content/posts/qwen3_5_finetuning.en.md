---
date: '2026-03-03T16:41:05+08:00'
draft: false
title: 'Fine-Tuning the Latest Qwen3.5 Model with LlamaFactory to Identify Humanoid Robot Models'
---

At the beginning of 2026, from the Consumer Electronics Show (CES) in Las Vegas to China’s Spring Festival Gala, domestically developed humanoid robots from Chinese companies have repeatedly broken into the mainstream spotlight. Their products and real-world applications have sparked heated discussions in overseas industries and continuously trended across global social media platforms and international media outlets.

Embodied intelligence is widely regarded as the next stage of artificial intelligence development. Its core lies in deeply coupling the intelligent “brain” with a physical “body,” directly transforming data, algorithms, and computing power into actionable capabilities that can reshape the physical world. Humanoid robots, due to their human-like appearance and functionality, are considered the advanced form and ideal carrier of embodied intelligence. They are expected to become the next generation of “super terminals,” following smartphones and new energy vehicles.

[LlamaFactory](https://github.com/hiyouga/LlamaFactory) is an open-source, low-code large language model fine-tuning framework. It integrates the most widely used fine-tuning techniques in the industry and supports zero-code fine-tuning through a Web UI interface. It has become one of the most popular fine-tuning frameworks in the open-source community, with nearly 70,000 GitHub stars.

The Tongyi Qianwen team has open-sourced the new-generation multimodal large model Qwen3.5. In this tutorial, we focus on how to fine-tune the open-source Qwen3.5-9B model using LlamaFactory for a specific task: **identifying humanoid robot models**. Through this hands-on practice, we aim to demonstrate how lightweight large models can empower embodied intelligence applications—enabling robots not only to “see,” but also to truly “understand.” We hope this serves as a practical contribution from the open-source community to the intelligent revolution sweeping the globe.

## Environment Requirements

- Recommended GPU memory: **≥ 32 GB**

## 1. Install LlamaFactory

Clone LlamaFactory to your local machine:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
````

Install dependencies:

```bash
pip install -e .
```

Run the following command. If the LlamaFactory version is displayed, the installation was successful:

```bash
llamafactory-cli version
```

## 2. Prepare the Dataset

[Manus](https://manus.im/app) is a general-purpose AI agent designed for complex task execution. It can autonomously complete end-to-end tasks from planning to execution. We use Manus to automatically construct a data crawling workflow, which is significantly more efficient than traditional web scraping scripts.

For example, the following prompt can be used to build the dataset:

```
I want to fine-tune a Qwen3.5 model that can identify all humanoid robot models appearing in the 2026 Spring Festival Gala. Please generate an automated dataset crawling workflow to collect images and descriptive information (such as model names, manufacturers, and application scenarios) of robots from the 2026 Year of the Horse Spring Festival Gala, as well as commonly seen humanoid robots domestically and internationally. 

Finally, provide a JSON dataset in the following format:
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Please identify and describe the robot in the image, paying special attention to its model, manufacturer, and application scenarios."
      },
      {
        "role": "assistant",
        "content": "The image shows the Unitree H1 robot produced by Unitree Robotics. It is equipped with advanced vision and depth sensor systems, enabling strong environmental perception capabilities. It can construct real-time 3D maps for autonomous navigation and obstacle avoidance."
      }
    ],
    "images": [
      "images/unitree_h1_5.png"
    ]
  },
  ...
]

Each robot image should correspond to one question and one answer (one data entry). Even if the robot model is the same, I want diverse question and answer styles.

The final dataset should contain at least 200 robot images. Provide both a Chinese JSON dataset and an English JSON dataset, along with the corresponding image files.
```

This tutorial provides a multi-turn dialogue dataset available at: [mllm_robot.zip](https://drive.google.com/file/d/1Bapd-3B_x8OW9mFOKiHTjG2GchKUBdUg/view?usp=sharing)

The dataset consists of 405 single-turn dialogue samples. Each sample includes one user instruction and one model response. During fine-tuning, the model continuously learns the response style from these samples to achieve robot identification capability.

Example:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Please identify and describe the robot in the image, focusing on its model, manufacturer, and application scenarios."
      },
      {
        "role": "assistant",
        "content": "The image shows the Unitree G1 (EDU) robot produced by Unitree Robotics. This cost-effective humanoid robot stands out for its excellent locomotion performance and compact design. It supports flexible folding for convenient storage and can perform complex movements such as jumping in place, demonstrating advanced dynamic control capabilities. The G1 (EDU) is designed as an open and powerful platform for research and education, promoting innovation and talent cultivation in robotics."
      }
    ],
    "images": [
      "images/unitree_g1_1.png"
    ]
  }
]
```

Download the dataset into the `LlamaFactory/data` directory and modify `dataset_info.json` by adding:

```json
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
  }
```

This allows LlamaFactory to recognize the newly added datasets.

## 3. Model Fine-Tuning

### 3.1 Launch the Web UI

After completing the preparation steps, run:

```bash
llamafactory-cli webui
```

Open the returned URL to access the Web UI.

### 3.2 Configure Parameters

After entering the Web UI, switch the language if needed.

* Model: **Qwen3.5-9B**
* Fine-tuning method: **LoRA**
* Dataset: `mllm_robot` and `mllm_robot_en`
* Learning rate: `1e-4`
* Epochs: `5`

### 3.3 Start Fine-Tuning

Set the output directory to:

```
train_qwen3_5_9B
```

The trained model weights will be saved in this directory.

Click “Preview Command” to display all configured parameters. If you prefer running fine-tuning via command line, copy and execute the generated command.

After starting fine-tuning, wait for model download and training to complete. You can monitor training progress and loss curves in the interface. On an RTX 5090, fine-tuning takes approximately 30 minutes. When “Finished” appears, the process is complete.

## 4. Model Chat

### 4.1 Chat with the Fine-Tuned Model

Go to the **Chat** tab.

* Set the checkpoint path to `train_qwen3_5_9B`
* Click “Load Model”

You can now interact with the fine-tuned model in the Web UI.

Upload a random robot image and ask the model to identify it.

The model correctly identifies the robot as the **MagicBot Z1 (2026 Spring Festival Gala Customized Edition)** designed by Magic Atom, indicating strong fine-tuning performance.

### 4.2 Chat with the Original Model

Click “Unload Model.”
Uncheck the checkpoint path and click “Load Model” again to interact with the original base model.

The base model fails to recognize the robot and mistakenly assumes it is a human performer in costume. This demonstrates that fine-tuning is effective.

## 5. Conclusion

In this tutorial, we demonstrated how to use Manus and LlamaFactory to fine-tune the Qwen3.5-9B model with LoRA, enabling it to identify humanoid robot models. Through manual testing, we verified the effectiveness of fine-tuning.

In future practice, real business datasets can be used for domain-specific fine-tuning, enabling the development of local multimodal large models capable of solving practical business challenges.
