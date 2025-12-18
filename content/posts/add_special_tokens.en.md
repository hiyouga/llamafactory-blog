---
date: '2025-12-17T14:13:17+08:00'
draft: false
title: 'Add New Special Tokens for Model Training'
---

## 1 Introduction

This paper uses the **Ministral-3-3B-Instruct-2512** model and takes an image classification task fine-tuned via **SFT** as an example to illustrate how to add new special tokens. The experimental command is as follows:

```bash
# install newest transformers
pip install git+https://github.com/huggingface/transformers

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=7 python src/train.py examples/train_lora/ministral3_lora_sft.yaml
```

It is necessary to preconfigure `ministral3_lora_sft.yaml`.

## 2 Dataset Loading and Preprocessing

In the file
[LLaMA-Factory/src/llamafactory/data/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276),
the **get_dataset** function is responsible for loading the dataset and preprocessing the data using the tokenizer.

### 2.1 Data Loading

The following code is part of the
[LLaMA-Factory/src/llamafactory/data/loader.py:get_dataset](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276)
function. It handles reading the data and converting it into the required format.

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

The loaded data are stored in `dataset`, and the data format is transformed as follows, for example:

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

### 2.2 Data Preprocessing

The data preprocessing code is located in
[LLaMA-Factory/src/llamafactory/data/loader.py:get_dataset](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L313), as shown below:

```python
with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
    dataset = _get_preprocessed_dataset(
        dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
    )
```

**This code converts data in `json` format into formatted sequence data**, for example:

```
'_prompt': [{'role': 'user', 'content': 'Transform the following sentence using a synonym: The car sped quickly.'}]
```

is converted to

```
'<|im_start|>user\nTransform the following sentence using a synonym: The car sped quickly.<|im_end|>\n<|im_start|>assistant\n'
```

Then, the sequence is converted into token IDs, and the function call flow is as follows:

`_get_preprocessed_dataset` $\rightarrow$ `SupervisedDatasetProcessor.preprocess_dataset` $\rightarrow$ `SupervisedDatasetProcessor._encode_data_example` $\rightarrow$ `SupervisedDatasetProcessor.template.encode_multiturn` $\rightarrow$ `Template._encode`

[Template._encode](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/template.py#L130) performs the conversion from sequences to token IDs. The code is as follows:

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

This function first performs format conversion to obtain `elements`, and then uses the `tokenizer` to convert `elements` into `token IDs`.

## 3 Special Tokens Parameter Passing

Adding special tokens requires using the `add_special_tokens` interface of the `tokenizer`, for example:

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

Therefore, to add special tokens in LLaMA-Factory, the required special tokens must be added to the tokenizer.

### 3.1 Tokenizer Loading Method

In `run_sft` under
[LLaMA-Factory/src/llamafactory/train/sft/workflow.py](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/train/sft/workflow.py#L41),
the tokenizer is loaded.

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

The function call path is:
[load_tokenizer](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/model/loader.py#L72) →
[patch_tokenizer](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/model/patcher.py#L60).

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

It can be seen that if `model_args` contains the `add_special_tokens` parameter, the corresponding special tokens will be loaded.

### 3.2 Model Arguments Loading Method

Now that we understand how the tokenizer is loaded, the key question becomes how `model_args` and its internal `add_special_tokens` are loaded.

In [_training_function](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/train/tuner.py#L52) under
[LLaMA-Factory/src/llamafactory/train/tuner.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/tuner.py),
the function reads the model arguments, data arguments, training arguments, and so on.

```python
def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    ......
```

The definition of
[get_train_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L253)
is as follows:

```python
def get_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False
    ......
```

Then it calls
[_parse_train_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L208),
which is defined as follows:

```python
def _parse_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)
```

Finally, it calls
[_parse_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L85),
which is defined as follows:

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

`parser: "HfArgumentParser"` parses all parameters defined in `_TRAIN_ARGS` within
`parser = HfArgumentParser(_TRAIN_ARGS)`, including `model_args`.

## 4 Example: Adding Special Tokens

### 4.1 Add them directly in the YAML file.

To add special tokens, you only need to include the `add_special_tokens` parameter in the training configuration file, for example:

```yaml
### model
model_name_or_path: Qwen2.5-3B-Instruct
trust_remote_code: true
add_special_tokens: "[start],[end]"
...
```

### 4.2 Configure the `new_special_tokens_config` file parameter.

A separate `new_special_tokens_config.yaml` file is required, for example:

```yaml
# SVG Container Tags
"<|START_OF_SVG|>": "Marks the beginning of an SVG document"
"<|END_OF_SVG|>": "Marks the end of an SVG document"

# SVG Group Tags
"<|start_of_g|>": "Begins a group element in SVG for organizing related shapes"
"<|end_of_g|>": "Ends a group element"
```

In this file, both the special tokens and their corresponding descriptions need to be defined.

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

`new_special_tokens_config` specifies the path to the `tokens_config.yaml` file, while `init_special_tokens` configures the method used to initialize the embeddings of the special tokens. The available options for `init_special_tokens` are `desc_init` and `desc_init_w_noise`. Initialization methods that leverage token descriptions allow the tokenizer to initialize token embeddings based on their descriptions.

**Note: Loading special tokens from a file takes higher priority than specifying special tokens directly in the configuration file.**

### 4.3 Adding via the Graphical User Interface

![image-20251217151544496](https://github.com/user-attachments/assets/bdb2e719-93b7-468e-9eb4-322507894279)

Simply add the content that would normally be specified in the YAML file under **Extra arguments**; **this method is equivalent to adding it directly in the YAML file**.

## 5 Validating Special Tokens

Here, a Pokémon image classification task is used to verify whether the special tokens can be correctly added, and to perform training and inference.

### 5.1 Preparing the Dataset

```python
from huggingface_hub import snapshot_download

repo_id = "fcakyon/pokemon-classification"
local_dir = "./pokemon-classification"

snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
print("Done！")
```

Use the script above to download the dataset.

Unzip the `train.zip` file under `pokemon-classification/data`, then use the script below to generate a JSON file adapted for LLaMA-Factory for training.

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

The resulting JSON file has the following format:

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

### 5.2 Training the Pokémon Multimodal Classification Model

- Registering the Dataset

Copy the generated dataset JSON file and the corresponding `train` folder into `LLaMA-Factory/data`. Then, add the following configuration to the `LLaMA-Factory/data/dataset_info.json` file to register the dataset:

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

- Training the Model with Special Tokens

```bash
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=7 USE_MODELSCOPE_HUB=1 llamafactory-cli webui
```

The special tokens used in this task are the names of Pokémon, and **add_special_tokens** needs to be added under Extra arguments.

```
"add_special_tokens":"[Dratini],[Kabuto],[Articuno],[Farfetchd],[Parasect],[Alolan Sandslash],[Gloom],[Jynx],[Muk],[Mew],[Machamp],[Eevee],[Doduo],[Kingler],[Kakuna],[MrMime],[Ninetales],[Golem],[Gyarados],[Dragonite]"
```

![image-20251218142836625](https://github.com/user-attachments/assets/032a98cc-9e2c-4621-bd26-21f0636e2862)

Once added, training can be started.

![image-20251218103414340](https://github.com/user-attachments/assets/eb0c9970-ca2c-494d-aff3-207c8d85c7c2)

### 5.3 Inference Using the Model

Similarly, **"add_special_tokens"** needs to be added under Extra arguments.

![image-20251218143111817](https://github.com/user-attachments/assets/0299db41-68e9-4859-821a-65b4953f00e2)

Input an image for classification. **Since the classification labels are special tokens, be sure to uncheck "Skip special tokens"**.

![image-20251218143421085](https://github.com/user-attachments/assets/3f65305d-baa1-48a7-94e3-490f2f0ac214)

The results from the original model are as follows:

![image-20251218105328875](https://github.com/user-attachments/assets/793a6b10-9da3-44e2-b3b8-f411cb80e441)

This indicates that the model has been properly trained and the special tokens have been successfully learned.