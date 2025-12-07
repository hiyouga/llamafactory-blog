---
date: '2025-04-03T21:13:00+08:00'
draft: false
title: 'Easy Dataset × LLaMA Factory: Empowering Large Models with Efficient Domain Knowledge Learning'
---

## 1 Introduction

[Easy Dataset](https://github.com/ConardLi/easy-dataset?utm_source=chatgpt.com) is an application designed specifically for creating fine-tuning datasets for large language models (LLMs). It provides an intuitive interface for uploading domain-specific documents, intelligently segmenting content, generating questions, and producing high-quality training data for model fine-tuning. It supports calling large models through APIs such as OpenAI, DeepSeek, Volcano Engine, as well as local models via Ollama.

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) is an open-source, low-code fine-tuning framework for large language models. It integrates the most widely used fine-tuning techniques in the industry and supports zero-code model fine-tuning through a Web UI. It has become one of the most popular fine-tuning frameworks in the open-source community, with over 63K stars on GitHub. It supports full-parameter fine-tuning, LoRA fine-tuning, as well as fine-tuning algorithms such as SFT and DPO.

This tutorial uses Easy Dataset to construct an SFT fine-tuning dataset from the publicly available financial reports of five internet companies and uses LLaMA Factory to fine-tune the Qwen2.5-3B-Instruct model, enabling the fine-tuned model to learn the knowledge contained in the financial report dataset.

## 2 System Requirements

- **GPU Memory:** ≥ 12 GB
- **CUDA Version:** above 11.6
- **Python Version:** 3.10

## 3 Generating Fine-Tuning Data with Easy Dataset

### 3.1 Installing Easy Dataset

#### Method 1: Using the Installation Package

If your operating system is Windows, macOS, or a Unix system with an ARM architecture, you can directly download the installation package from the Easy Dataset repository: https://github.com/ConardLi/easy-dataset/releases/latest .

#### Method 2: Using Dockerfile

1.Pulling the Easy Dataset Repository from GitHub

```Bash
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset
```

2.Building the Docker Image

```Bash
docker build -t easy-dataset .
```

3.Running the Container

```Bash
docker run -d \
    -p 1717:1717 \
    -v {YOUR_LOCAL_DB_PATH}:/app/local-db \
    --name easy-dataset \
    easy-dataset
```

Replace `{YOUR_LOCAL_DB_PATH}` with a local directory to serve as the mount path for `/app/local-db` in the Docker runtime. After starting, open the web interface at [http://localhost:1717](http://localhost:1717/) to use Easy Dataset’s UI.

#### Method 3: Using NPM

1.Download Node.js and pnpm

Visit the official websites to install Node.js and pnpm: https://nodejs.org/en/download | https://pnpm.io/

Check that the Node.js version is above 18.0:

```bash
node -v  # v22.14.0
```

2.Clone the Easy Dataset Repository from GitHub

```bash
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset
```

3.Install Dependencies

```
pnpm install
```

4.Start the Easy Dataset Application

```bash
pnpm build
pnpm start
```

If the console shows the following output, it means the application has started successfully. Open your browser and visit http://localhost:1717 to access the Easy Dataset interface:

```
> easy-dataset@1.2.3 start
> next start -p 1717

  ▲ Next.js 14.2.25
  - Local:        http://localhost:1717

 ✓ Ready in 287ms
```

### 3.2 Sample Data Download

This tutorial provides a set of financial reports from internet companies as sample data, including the Q2 2024 reports of five domestic internet companies in TXT and Markdown formats. You can download them using Git or by directly visiting the [repository link](https://github.com/the-seeds/FinancialData-SecondQuarter-2024).

```bash
git clone https://github.com/the-seeds/FinancialData-SecondQuarter-2024.git
```

All data are in plain text format. Below is a sample excerpt.

```
快手二季度净利润增超七成，CEO程一笑强调可灵AI商业化

8月20日，快手科技发布2024年第二季度业绩，总营收同比增长11.6%至约310亿元，经调整净利润同比增长73.7%达46.8亿元左右。该季度，快手的毛利率和经调整净利润率均达到单季新高，分别为55.3%和15.1%。值得一提的是，针对今年加码的AI相关业务，快手联合创始人、董事长兼CEO程一笑在财报后的电话会议上表示，可灵AI将寻求更多与B端合作变现的可能性，也会探索将大模型进一步运用到商业化推荐中，提升算法推荐效率。

线上营销服务贡献近六成收入，短剧日活用户破3亿

财报显示，线上营销服务、直播和其他服务（含电商）收入依然是拉动快手营收的“三驾马车”，分别占总营收的56.5%、30.0%和13.5%。线上营销服务收入由2023年同期的143亿元增加22.1%至2024年第二季度的175亿元，财报解释主要是由于优化智能营销解决方案及先进的算法，推动营销客户投放消耗增加。
```

### 3.3 Fine-Tuning Data Generation

#### Create Project and Configure Parameters

1.After opening the Easy Dataset homepage in your browser, click **Create Project**.

![image-20251201154419126](https://github.com/user-attachments/assets/baf62040-0e0b-42d7-a18b-43fe66da920a)

2.First, enter the **Project Name** (required). The other two fields can be left blank. Then click **Create Project** to confirm.

![image-0001](https://github.com/user-attachments/assets/9e2b17f2-6779-4346-9ed0-cfb9b50ab6ca)

3.After the project is created, you will be redirected to the **Project Settings** page. Open **Model Configuration** and select the large model API to be used for data generation.

![image-20251201154717628](https://github.com/user-attachments/assets/6e8cb11c-4a81-4c0a-a47a-244ea01093b1)

4.Here, we use the DeepSeek model as an example. Enter the model **Provider** and **Model Name**, and provide the **API Key**. Click **Save** to store the data locally. Then, select the configured model from the top-right corner. The **API Key** must be obtained from the model provider and must be valid for accessing the provider’s large model.

![image-34244445](https://github.com/user-attachments/assets/8c8da760-aaa5-49ae-aaae-db72e5ff45ee)

5.Open the **Task Configuration** page and set the text segmentation length to a minimum of 500 characters and a maximum of 2000 characters. In the question generation settings, change it to generate one question per 10 characters. After making the changes, click **Save Task Configuration** at the bottom of the page.

![image-08728377](https://github.com/user-attachments/assets/adcae63f-cd7d-45e9-ac3f-397bc710a688)

#### **Process Data Files**

1.Open the **Document Processing** page and select a model.

![image-986387](https://github.com/user-attachments/assets/b5fb6270-290d-4ac5-9ff9-eb0ba3881992)

![image-86668](https://github.com/user-attachments/assets/fd95b8f3-58be-4909-ba41-44de2e99e01d)

2.After selecting the files, click **Upload and Process Files**.

![image-987976](https://github.com/user-attachments/assets/06cb119d-42e8-4172-a4ad-b3e6ad65e114)

3.After uploading, the large model will be used to parse the file content and segment it. Please wait patiently for the processing to complete. Sample data usually takes around 2 minutes.

![image-09173](https://github.com/user-attachments/assets/b620527e-53f9-4eb0-9ba8-edfeae60b320)

#### **Generate Fine-Tuning Data**

1.Once the file processing is complete, you can see the text segments after splitting. Select all the text segments and click **Auto Generate**.

![image-20251201160011221](https://github.com/user-attachments/assets/3d4a510f-1003-4427-8c87-4fa652abb9f2)

2.After clicking, the large model will be used to generate questions based on the text segments. Please wait patiently for the process to complete. Depending on the API speed, it usually takes around 2 minutes.

![image-20251201160154221](https://github.com/user-attachments/assets/e7d1ee9b-e4a2-4b43-8c34-12543231506e)

#### **Export Dataset to LLaMA Factory**

1.After all answers have been generated, open the **Dataset Management** page and click **Export Dataset**.

![image-20251201160342835](https://github.com/user-attachments/assets/4c6cf6c4-4d96-4e8e-886a-a02ae7273a2e)

You can see the task in progress in the background. Wait approximately 2 minutes for it to complete.

![image-20251201160436716](https://github.com/user-attachments/assets/bf3474f4-66d4-44f7-a244-6432f2a3b00f)

2.Export the Dataset on the Single-Turn QA Dataset Page

![image-20251201160857619](https://github.com/user-attachments/assets/e9dec884-63ce-406d-a417-c43d8e0377f7)

3.In the export configuration, select **Use in LLaMA Factory**, then click **Update LLaMA Factory Configuration**. This will generate a configuration file in the corresponding folder. Click the **Copy** button to copy the configuration path to the clipboard.

![image-971397](https://github.com/user-attachments/assets/67448a9f-b8c3-4985-8a08-2909b0391be4)

4.In the folder corresponding to the configuration file path, you can find the generated data files. The main files to focus on are:

a. **dataset_info.json**: The dataset configuration file required by LLaMA Factory  
b. **alpaca.json**: The dataset file organized in Alpaca format  
c. **sharegpt.json**: The dataset file organized in ShareGPT format   

Both the Alpaca and ShareGPT formats can be used for fine-tuning, and the contents of the two files are identical.

![image-1838818](https://github.com/user-attachments/assets/50f0d718-8b72-4ba8-b40d-181c37a359be)

## 4 Fine-Tune the Qwen2.5-3B-Instruct Model Using LLaMA Factory

### 4.1 Install LLaMA Factory

1.Create a Virtual Environment for the Experiment (Optional)

```bash
conda create -n llamafactory python=3.10
```

2.Clone the LLaMA Factory Repository from GitHub and Install Environment Dependencies

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,modelscope]"
```

3.Run `llamafactory-cli version` to verify. If the current LLaMA Factory version is displayed, the installation was successful.

```
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.2                |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

### 4.2 Start the fine-tuning task

1.After confirming that LLaMA Factory has been successfully installed, run the following command to launch LLaMA Board.

```bash
CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 llamafactory-cli webui
```

Environment variable explanation:

- **CUDA_VISIBLE_DEVICES**: Specifies the GPU device index to use. By default, all GPUs are used.
- **USE_MODELSCOPE_HUB**: Enables accelerated model downloads from the ModelScope Hub (China). Disabled by default.

After successful startup, the following information will appear in the console.
 Open **http://localhost:7860** in your browser to access the Web UI.

![image-20251201162731159](https://github.com/user-attachments/assets/85c8a3cf-0057-409f-983e-1adeda53d305)

2.After entering the Web UI interface, select the model **Qwen2.5-3B-Instruct**. You can specify the local absolute path for the model. If left blank, it will be downloaded from the internet.

![image-20251201193549815](https://github.com/user-attachments/assets/3589db0f-4a08-4c5c-833b-9f635b6686d8)

3.Set the **dataset path** to the configuration path exported by **Easy Dataset**, and select the **Alpaca** format dataset.

![image-20251201194656868](https://github.com/user-attachments/assets/40dd4be6-78bd-4924-b097-4fc15a400b27)

4.To help the model learn the dataset more effectively, set the **learning rate** to **1e-4** and increase the **number of training epochs** to **8**. The batch size and gradient accumulation should be adjusted according to the available GPU memory. If memory allows, increasing the batch size can speed up training. In general, ensure that **Batch Size × Gradient Accumulation × Number of GPUs = 32**.

![image-20251201194825884](https://github.com/user-attachments/assets/be99e73c-e892-4e79-b0c6-cd03d3c453bb)

5.Click on **Other Parameters**, and set the **save interval** to 50. Saving more checkpoints helps observe how the model’s performance changes over training epochs.

![image-20251201194941036](https://github.com/user-attachments/assets/25099eff-d560-40fd-93b5-3c6b7077526f)

6.Click **LoRA Parameter Settings**, set the **LoRA rank** to 16, and set the **LoRA scaling factor** to 32.

![image-20251201195017902](https://github.com/user-attachments/assets/6b6017a3-05ca-4c98-b4da-dd4f0b50b52b)

7.Click the **Start** button, wait for the model to download, and after some time you should be able to observe the loss curve during training.

![image-20251201195713189](https://github.com/user-attachments/assets/315c7d9b-947f-449b-ae68-895b88e1bd16)

8.Wait for the model training to complete. Depending on GPU performance, the training time may range from 20 to 60 minutes.

### 4.3 Validate Fine-Tuning Results

1.Select the **Checkpoint Path** as the output directory from earlier, open the **Chat** page, and click **Load Model**.

![image-20251201200151409](https://github.com/user-attachments/assets/ec2d71ef-dad4-4c07-8cd0-9a291e38dc11)

2.Enter your question in the chat box below and click **Submit** to interact with the model. Comparing with the original data, the fine-tuned model provides correct answers.

![image-20251201200339699](https://github.com/user-attachments/assets/4ac3529c-e9dd-46e1-ba6a-fe598e18e42e)

3.Click **Unload Model** to unload the fine-tuned model. Clear the **Checkpoint Path** and click **Load Model** to load the original pre-trained model.

![image-20251201200424199](https://github.com/user-attachments/assets/38c11e8c-360a-469d-ad7f-cc39604592dc)

4.Enter the same question and interact with the model. You will find that the original model answers incorrectly, which demonstrates that the fine-tuning was effective.

![image(10)](https://github.com/user-attachments/assets/1bad8716-5c40-4caf-9d64-35cfb2f55425)The fine-tuning effect of the 3B model is relatively limited and is used here only for tutorial demonstration.
 For better results, it is recommended to try the 7B or 14B models when sufficient resources are available.

You are welcome to follow the GitHub repository:

- Easy Dataset: https://github.com/ConardLi/easy-dataset
- LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory
