---
date: '2026-03-03T16:41:05+08:00'
draft: false
title: 'Fine-Tuning the Latest Qwen3.5 Model to Identify Humanoid Robot Models Using LlamaFactory'
---

At the beginning of 2026, from the Consumer Electronics Show (CES) in Las Vegas, USA, to the China Central Television (CCTV) Spring Festival Gala, China's self-developed humanoid robots have frequently "broken through the circle." Products and applications from multiple Chinese enterprises have not only sparked discussions within the overseas industry but have also continuously "swept" global social media platforms and international media. Embodied intelligence, regarded as the next stage of artificial intelligence development, has its core in achieving a deep coupling between the intelligent "brain" and the physical "body," thereby directly transforming data, algorithms, and computing power into the ability to act on and transform the objective world. Humanoid robots, due to their human-like appearance and functionality, are considered a high-level form and the optimal carrier for embodied intelligence, poised to become the next-generation super terminal following smartphones and new energy vehicles.

[LlamaFactory](https://github.com/hiyouga/LlamaFactory) is an open-source, low-code large model fine-tuning framework. It integrates the most widely used fine-tuning techniques in the industry and supports zero-code fine-tuning of large models through a Web UI interface. It has now become one of the most popular fine-tuning frameworks in the open-source community, with nearly 70,000 GitHub stars .

The Tongyi Qianwen team has open-sourced the new-generation multimodal large model Qwen3.5. This tutorial will focus on how to use the open-source Qwen3.5-9B model, leveraging the LlamaFactory open-source low-code large model fine-tuning framework, to fine-tune for the specific task of "identifying humanoid robot models." Through this practice, we aim to demonstrate how lightweight large models can empower embodied intelligence applications, enabling robots to not only "see" but also "understand," thereby contributing a practical force from the open-source community to this intelligent revolution that is sweeping the globe.

## Runtime Environment Requirements
- It is recommended to have a GPU with at least 32 GB of video memory.

## 1. Install LlamaFactory

Clone LlamaFactory to your local machine:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
```

Install the LlamaFactory environment dependencies:

```bash
pip install -e .
```

Run the following command. If the LlamaFactory version is displayed, the installation was successful.

```bash
llamafactory-cli version
```

## 2. Prepare the Dataset

[Manus](https://manus.im/app) is a general-purpose AI agent focused on executing complex tasks, capable of autonomously completing end-to-end tasks from planning to execution. We used Manus to automate the construction of a data scraping workflow, which is highly efficient compared to traditional methods like writing crawler scripts. For example, you can use the following prompt to accomplish dataset acquisition:

```
I want to fine-tune a Qwen3.5 model that can identify all humanoid robot models appearing in the 2026 Spring Festival Gala. I hope you will generate an automated dataset scraping workflow for me, crawling images of robots from the 2026 Year of the Horse Spring Festival Gala as well as common domestic and foreign humanoid robots, along with their model information, manufacturers, and other descriptive details. Then, finally, give me a JSON dataset in the following format:
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Please identify and describe the robot in the image and its features, paying special attention to its model, manufacturer, and application scenarios."
      },
      {
        "role": "assistant",
        "content": "The image shows the Unitree H1 robot produced by Unitree Robotics. It is equipped with an advanced vision and depth sensor system, giving it powerful environmental perception capabilities, enabling it to build 3D maps in real-time and achieve autonomous navigation and obstacle avoidance."
      }
    ],
    "images": [
      "images/unitree_h1_5.png"
    ]
  },
  ...
]
I hope each robot image corresponds to one question and answer (one data entry), and the user's question and the assistant's answer should not be monotonous. Even if the robot model is the same, I hope for diversity in the questions and answers.
Ultimately, I hope to obtain a dataset with at least 200 robot images, giving me a Chinese JSON dataset, an English JSON dataset, and the corresponding image files.
```

This tutorial provides a multi-turn conversation dataset. The link is: [mllm_robot.zip](https://drive.google.com/file/d/1Bapd-3B_x8OW9mFOKiHTjG2GchKUBdUg/view?usp=sharing). The samples in the dataset are in a single-turn conversation format, containing 405 samples. Each sample consists of one user instruction and one model response. During the fine-tuning process, the model continuously learns the response style from the samples, thereby achieving the goal of identifying robots. A sample of the data is shown below:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Please identify and describe the robot in the image and its features, paying special attention to its model, manufacturer, and application scenarios."
      },
      {
        "role": "assistant",
        "content": "The image shows the Unitree G1 (EDU) robot produced by Unitree Robotics. This cost-effective humanoid robot stands out with its excellent motion performance and compact design. It not only supports flexible folding for storage, greatly saving space, but also can complete complex actions like jumping in place, demonstrating its advanced dynamic control capabilities. The G1 (EDU) aims to provide an open and powerful platform for the research and education sectors, fostering the development of robotics technology and the cultivation of innovative talent."
      }
    ],
    "images": [
      "images/unitree_g1_1.png"
    ]
  }
]
```

You can download this dataset and place it under `LlamaFactory/data`, and modify the `dataset_info.json` file by adding the following content:

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

This allows LlamaFactory to recognize the newly added dataset.

## 3. Model Fine-Tuning

### 3.1 Launch the Web UI

After completing the preliminary preparations, you can launch the Web UI by running the following command:

```bash
llamafactory-cli webui
```

Click on the returned URL address to enter the Web UI page.

### 3.2 Configure Parameters

After entering the WebUI, you can switch the language according to your needs. First, configure the model. This tutorial selects the **Qwen3.5-9B** model, and the fine-tuning method is changed to **lora**.

![image-20260303152428178](https://github.com/user-attachments/assets/f6eb4c49-7cdb-406c-8d1a-e6963d74a90f)

For the dataset, use `mllm_robot` and `mllm_robot_en`. Use a learning rate of `1e-4`, and set Epochs to 5.

![image-20260303152509242](https://github.com/user-attachments/assets/884a102a-b507-464f-aaae-db44a1fbfbd8)

### 3.3 Start Fine-Tuning

Change the output directory to `train_qwen3_5_9B`, where the trained model weights will be saved. Clicking "Preview Command" will display all configured parameters. If you wish to run the fine-tuning via code, you can copy this command and run it in the command line.

![image-20260303152540908](https://github.com/user-attachments/assets/d5c6cd90-4b7e-4978-8f5f-a4998844374b)

After starting fine-tuning, you need to wait for some time. After the model is downloaded, you can observe the training progress and loss curve in the interface. On an RTX 5090, model fine-tuning takes approximately 30 minutes. The message "Training Finished" indicates successful fine-tuning.

![image-20260303160358571](https://github.com/user-attachments/assets/e36aa989-57f2-4886-84ae-d9d66c30d533)

## 4. Model Dialogue

### 4.1 Dialogue with the Fine-Tuned Model

Select the "Chat" tab. Change the **Checkpoint Path** to `train_qwen3_5_9B`, and click "Load Model" to start a dialogue with the fine-tuned model in the Web UI.

![image-20260303161002113](https://github.com/user-attachments/assets/6a75d0e1-3fea-4b64-9d86-e65caa5026ba)

Randomly upload an image and let the model identify the robot in the image.

![image-20260303161632573](https://github.com/user-attachments/assets/1ffad2d9-18cb-4db2-b8bd-6436bd75ffa1)

The model correctly identified the robot in the image as the MagicBot Z1 (2026 Spring Festival Gala Custom Edition) robot designed by MagicLab, indicating a good fine-tuning effect .

### 4.2 Dialogue with the Original Model

Click "Unload Model," then **uncheck** the checkpoint path input box, and click "Load Model" again to chat with the original model before fine-tuning.

![image-20260303164535031](https://github.com/user-attachments/assets/4eca83d5-d1dd-490a-90c9-49751b365d75)

The model did not identify the robot in the image and instead thought the robot was a person in a costume. This demonstrates that the model fine-tuning was effective.

## 5. Summary

This tutorial introduced how to use the Manus and LlamaFactory frameworks to fine-tune the Qwen3.5-9B model using LoRA, enabling it to identify robot models. The fine-tuning effect was verified through manual testing. In subsequent practices, you can use actual business datasets to fine-tune the model, obtaining a local domain-specific multimodal large model capable of solving problems in actual business scenarios.
