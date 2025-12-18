---
date: '2025-12-17T14:13:17+08:00'
draft: false
title: '添加 Special Tokens 训练模型'
---

## 1 引言

本文使用 Ministral-3-3B-Instruct-2512 模型通过 SFT 一个图像分类任务为例来介绍如何添加新的 special tokens。实验的运行命令为：

```bash
# install newest transformers
pip install git+https://github.com/huggingface/transformers

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=7 python src/train.py examples/train_lora/ministral3_lora_sft.yaml
```

需要预先配置好 `ministral3_lora_sft.yaml`。

## 2 数据集加载和预处理

[LLaMA-Factory/src/llamafactory/data/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276) 这个文件下的 [**get_dataset**](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276) 函数完成数据集的加载，并且使用 tokenizer 预处理数据。

### 2.1 数据加载

下面的代码是 [LLaMA-Factory/src/llamafactory/data/loader.py:get_dataset](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276) 函数的一部分，完成数据的读取并且转换数据格式。

```python
# Load and preprocess dataset
with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
    dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
    eval_dataset = _get_merged_dataset(
        data_args.eval_dataset,
        model_args,
        data_args,
        training_args,
        stage,
        return_dict=data_args.eval_on_each_dataset,
    )
```

加载的数据放在 `dataset` 里面，并且格式转变为如下，例如：

```json
[
    {
        '_prompt': [{'role': 'user', 'content': 'Transform the following sentence using a synonym: The car sped quickly.'}],
        '_response': [{'role': 'assistant', 'content': 'The car accelerated rapidly.'}],
        '_system': '',
        '_tools': '',
        '_images': None, 
        '_videos': None, 
        '_audios': None
    }
]
```

### 2.2 数据预处理

数据预处理的代码位于 [LLaMA-Factory/src/llamafactory/data/loader.py:get_dataset](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L313)，如下：

```python
with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
    dataset = _get_preprocessed_dataset(
        dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
    )
```

**这段代码完成 `json` 格式数据向格式化序列数据转换**，例如：

```
'_prompt': [{'role': 'user', 'content': 'Transform the following sentence using a synonym: The car sped quickly.'}]
```

转为

```
'<|im_start|>user\nTransform the following sentence using a synonym: The car sped quickly.<|im_end|>\n<|im_start|>assistant\n'
```

然后完成序列到 token ID 的转换，函数调用流程如下：

`_get_preprocessed_dataset` $\rightarrow$ `SupervisedDatasetProcessor.preprocess_dataset` $\rightarrow$ `SupervisedDatasetProcessor._encode_data_example` $\rightarrow$ `SupervisedDatasetProcessor.template.encode_multiturn` $\rightarrow$ `Template._encode`

[Template._encode](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/template.py#L130) 完成序列到 token ID 的转换，代码如下：

```python
def _encode(
    self,
    tokenizer: "PreTrainedTokenizer",
    messages: list[dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
) -> list[list[int]]:
    r"""Encode formatted inputs to pairs of token ids.

    Turn 0: prefix + system + query        resp
    Turn t: query                          resp.
    """
    system = system or self.default_system
    encoded_messages = []
    for i, message in enumerate(messages):
        elements = []

        if i == 0:
            elements += self.format_prefix.apply()
            if system or tools:
                tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                elements += self.format_system.apply(content=(system + tool_text))

        if message["role"] == Role.USER:
            elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
        elif message["role"] == Role.ASSISTANT:
            elements += self.format_assistant.apply(content=message["content"])
        elif message["role"] == Role.OBSERVATION:
            elements += self.format_observation.apply(content=message["content"])
        elif message["role"] == Role.FUNCTION:
            elements += self.format_function.apply(
                content=message["content"], thought_words=self.thought_words, tool_call_words=self.tool_call_words
            )
        else:
            raise NotImplementedError("Unexpected role: {}".format(message["role"]))

        encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

    return encoded_messages
```

这个函数首先完成格式转换得到 `elements`，然后使用 `tokenizer` 将 `elements` 转换为 `token ids` 。

## 3 Special Tokens 参数传递

添加 Special Tokens 需要使用 `tokenizer` 的 `add_special_tokens` 接口，例如：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

special_tokens_dict = {
    "additional_special_tokens": [
        "<start>",
        "<end>",
    ]
}

num_added = tokenizer.add_special_tokens(special_tokens_dict)
print("Added tokens:", num_added)
```

因此，想要在 LLaMA-Factory 里面添加 Special Tokens，就需要将所需添加的 Special Tokens 添加到 tokenizer 里面。

### 3.1 tokenizer 加载方法

在 [LLaMA-Factory/src/llamafactory/train/sft/workflow.py](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/train/sft/workflow.py#L41) 下的 `run_sft` 里面加载了 tokenizer

```python
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    ......
```

函数调用路径为：[load_tokenizer](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/model/loader.py#L72) $\rightarrow$ [patch_tokenizer](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/model/patcher.py#L60)

```python
def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    ......

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")
```

可以看到，如果 `model_args` 有 `add_special_tokens` 这个参数，则会加载 `add_special_tokens`。

### 3.2 model_args 加载方法

知道了 tokenizer 是如何加载的，现在关键的问题是如何加载 `model_args` 及其内部的 `add_special_tokens`。

在 [LLaMA-Factory/src/llamafactory/train/tuner.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/tuner.py) 下的 [_training_function](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/train/tuner.py#L52) 函数读取了模型参数，数据参数，训练参数等。

```python
def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    ......
```

其中 [get_train_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L253) 的定义如下：

```python
def get_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False
    ......
```

然后需要调用 [_parse_train_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L208) ，其定义如下：

```python
def _parse_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)
```

最终需要调用 [_parse_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L85)，其定义如下：

```python
def _parse_args(
    parser: "HfArgumentParser", args: Optional[Union[dict[str, Any], list[str]]] = None, allow_extra_keys: bool = False
) -> tuple[Any]:
    args = read_args(args)
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args)
```

`parser: "HfArgumentParser"` 会解析所有 `parser = HfArgumentParser(_TRAIN_ARGS)` 中 `_TRAIN_ARGS` 定义的参数，包括 `model_args` 。

## 4 添加 Special Tokens 示例

### 4.1 直接在 yaml 文件里面添加

添加 special tokens 只需要在训练配置文件里面添加 `add_special_tokens` 参数，例如：

```yaml
### model
model_name_or_path: Qwen2.5-3B-Instruct
trust_remote_code: true
add_special_tokens: "[start],[end]"
...
```

### 4.2 配置 new_special_tokens_config 文件参数添加

需要一个独立的 new_special_tokens_config.yaml 文件，例如：

```yaml
# SVG Container Tags
"<|START_OF_SVG|>": "Marks the beginning of an SVG document"
"<|END_OF_SVG|>": "Marks the end of an SVG document"

# SVG Group Tags
"<|start_of_g|>": "Begins a group element in SVG for organizing related shapes"
"<|end_of_g|>": "Ends a group element"
```

在这个文件里面需要同时定义 special tokens 和其对应的描述。

```bash
### model
model_name_or_path: Qwen2.5-3B-Instruct
trust_remote_code: true
...

# Training config
new_special_tokens_config: examples/extras/multi_tokens/tokens_cfg.yaml
init_special_tokens: desc_init
...

# Inference config
skip_special_tokens: false  # Must set to false for structured tokens
...
```

`new_special_tokens_config` 指示 tokens_config.yaml 文件路径，`init_special_tokens` 配置 special tokens 初始化 embedding 的方法，`init_special_tokens` 可选 `desc_init` 和 `desc_init_w_noise` 。具备 token 描述的初始化方法可以使 tokenizer 通过 token 描述初始化 token 的 embedding。

**注意：通过文件的形式加载 special tokens 比通过直接在配置文件上指定 special tokens 的优先级更高。**

### 4.3 可视化界面添加

![image-20251217151544496](https://github.com/user-attachments/assets/bdb2e719-93b7-468e-9eb4-322507894279)

在 `Extra arguments` 下面添加原本需要在 yaml 文件下添加的内容即可，**这种添加方式和直接在 yaml 文件添加等价**。

## 5 验证 Special Token

这里使用宝可梦图片分类任务验证 special token 是否可以正确添加，并训练和推理。

### 5.1 准备数据集

```python
from huggingface_hub import snapshot_download

repo_id = "fcakyon/pokemon-classification"
local_dir = "./pokemon-classification"

snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
print("Done！")
```

使用上面的脚本下载数据集。

解压 `pokemon-classification/data` 下面的 train.zip 文件，使用下面的脚本生成 LLaMA-Factory 适配的 json 文件用于训练。

```python
import os
import json

train_dir = "train"
output_file = "pokemon_dataset.json"

dataset = []

special_tokens_list = []

for class_name in os.listdir(train_dir)[:20]:
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    special_tokens_list.append(class_name)

    for img_file in os.listdir(class_path):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        img_path = os.path.join(class_path, img_file)

        data_item = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>Who is this Pokemon?"
                },
                {
                    "role": "assistant",
                    "content": f"[{class_name}]"
                },
                {
                    "role": "user",
                    "content": "What type is it?<image>"
                },
                {
                    "role": "assistant",
                    "content": f"[{class_name}]"
                }
            ],
            "images": [
                img_path,
                img_path
            ]
        }

        dataset.append(data_item)

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generation completed. A total of {len(dataset)} data entries were generated and saved to {output_file}.")
special_tokens = ""
for token in special_tokens_list:
    special_tokens += f"[{token}],"
print(f"special_tokens: {special_tokens}.")

```

得到的 json 文件格式如下：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Who is this Pokemon?"
      },
      {
        "role": "assistant",
        "content": "[Dratini]"
      },
      {
        "role": "user",
        "content": "What type is it?<image>"
      },
      {
        "role": "assistant",
        "content": "[Dratini]"
      }
    ],
    "images": [
      "train/Dratini/d767470f6a6e44f6b3076282d4d416cf_jpg.rf.0d1a118bbc525e1772ace46ea075ca1e.jpg",
      "train/Dratini/d767470f6a6e44f6b3076282d4d416cf_jpg.rf.0d1a118bbc525e1772ace46ea075ca1e.jpg"
    ]
  }
]
```

### 5.2 训练 Pokemon 多模态分类模型

- 注册数据集

把生成的数据集 json 文件和对应的 train 文件夹拷贝到 `LLaMA-Factory/data` 下。然后在 `LLaMA-Factory/data/dataset_info.json` 文件里面加上如下配置用于注册数据集：

```json
"pokemon_dataset": {
    "file_name": "pokemon_dataset.json",
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

- 添加 Special Token 训练模型

```bash
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=7 USE_MODELSCOPE_HUB=1 llamafactory-cli webui
```

本任务使用的 special token 是宝可梦的名字，需要在 Extra arguments 添加 **add_special_tokens** 。

```
"add_special_tokens":"[Dratini],[Kabuto],[Articuno],[Farfetchd],[Parasect],[Alolan Sandslash],[Gloom],[Jynx],[Muk],[Mew],[Machamp],[Eevee],[Doduo],[Kingler],[Kakuna],[MrMime],[Ninetales],[Golem],[Gyarados],[Dragonite]"
```

![image-20251218142836625](https://github.com/user-attachments/assets/032a98cc-9e2c-4621-bd26-21f0636e2862)添加完之后可以开始训练了。

![image-20251218103414340](https://github.com/user-attachments/assets/eb0c9970-ca2c-494d-aff3-207c8d85c7c2)

### 5.3 使用模型进行推理

同样需要在 Extra arguments 添加 **"add_special_tokens"**  

![image-20251218143111817](https://github.com/user-attachments/assets/0299db41-68e9-4859-821a-65b4953f00e2)

输入图片进行分类，**由于分类标签是 special token，一定要取消勾选 Skip special tokens**。

![image-20251218143421085](https://github.com/user-attachments/assets/3f65305d-baa1-48a7-94e3-490f2f0ac214)

原始模型的结果如下，

![image-20251218105328875](https://github.com/user-attachments/assets/793a6b10-9da3-44e2-b3b8-f411cb80e441)

说明模型被训练到位了，special tokens 被训练到位了。
