---
date: '2025-12-17T14:13:17+08:00'
draft: false
title: 'Add New Special Tokens for Model Training'
---

## 1 Introduction

This article uses the Qwen2.5-3B-Instruct model as an example with SFT to illustrate how to add new special tokens. The command to run the experiment is:

```bash
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train_lora/qwen2.5_lora_sft.yaml
```

You need to pre-configure the `qwen2.5_lora_sft.yaml` file.

## 2 Dataset Loading and Preprocessing

The [**get_dataset**](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276) function in the file [LLaMA-Factory/src/llamafactory/data/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276) handles the dataset loading and preprocesses the data using a tokenizer.

### 2.1 Data Loading

The following code is part of the [get_dataset](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L276) function in `LLaMA-Factory/src/llamafactory/data/loader.py`, responsible for reading the data and converting it into the required format.

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

The loaded data is stored in `dataset` and its format is converted as follows, for example:

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

The code for data preprocessing is located in [get_dataset](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/loader.py#L313) in `LLaMA-Factory/src/llamafactory/data/loader.py`, as shown below:

```python
with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
    dataset = _get_preprocessed_dataset(
        dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
    )
```

**This code converts data in `json` format into a structured, formatted dataset**, for example:

```
'_prompt': [{'role': 'user', 'content': 'Transform the following sentence using a synonym: The car sped quickly.'}]
```

Converted to

```
'<|im_start|>user\nTransform the following sentence using a synonym: The car sped quickly.<|im_end|>\n<|im_start|>assistant\n'
```

It also performs the tokenization process, with the function call flow as follows:

`_get_preprocessed_dataset` $\rightarrow$ `SupervisedDatasetProcessor.preprocess_dataset` $\rightarrow$ `SupervisedDatasetProcessor._encode_data_example` $\rightarrow$ `SupervisedDatasetProcessor.template.encode_multiturn` $\rightarrow$ `Template._encode`

[Template._encode](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/data/template.py#L130) handles the conversion from sequences to token IDs. The code is as follows:

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

This function first performs the format conversion and then uses the `tokenizer` to convert the `elements` into `token IDs`.

## 3 Passing Special Tokens Parameters

Adding special tokens requires using the `add_special_tokens` method of the `tokenizer`, for example:

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

Therefore, to add special tokens in LLaMA-Factory, you need to include the desired special tokens into the tokenizer.

### 3.1 Tokenizer Loading Method

The tokenizer is loaded within the `run_sft` function in [LLaMA-Factory/src/llamafactory/train/sft/workflow.py](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/train/sft/workflow.py#L41).

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

The function call path is: [load_tokenizer](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/model/loader.py#L72) → [patch_tokenizer](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/model/patcher.py#L60).

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

It can be seen that if `model_args` contains the `add_special_tokens` parameter, the specified special tokens will be loaded.

### 3.2 Method for Loading `model_args`

Now that we understand how the tokenizer is loaded, the key question is how `model_args` and its internal `add_special_tokens` are loaded.

The `_training_function` function in [LLaMA-Factory/src/llamafactory/train/tuner.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/tuner.py#L52) reads the model parameters, data parameters, training parameters, and more.

```python
def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    ......
```

The definition of [get_train_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L253) is as follows:

```python
def get_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False
    ......
```

Next, it calls [_parse_train_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L208), which is defined as follows:

```python
def _parse_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)
```

Finally, it calls [_parse_args](https://github.com/hiyouga/LLaMA-Factory/blob/9fd4b094d4adadd34bc46769d385d11870103ad7/src/llamafactory/hparams/parser.py#L85), which is defined as follows:

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

`parser: "HfArgumentParser"` parses all the parameters defined in `_TRAIN_ARGS` within `parser = HfArgumentParser(_TRAIN_ARGS)`, including `model_args`.

## 4 Example of Adding Special Tokens

### 4.1 Add Directly in the YAML File

To add special tokens, simply include the `add_special_tokens` parameter in the training configuration file, for example:

```yaml
### model
model_name_or_path: Qwen2.5-3B-Instruct
trust_remote_code: true
add_special_tokens: "[start],[end]"
...
```

### 4.2 Add Special Tokens via `new_special_tokens_config` File Parameter

A separate `new_special_tokens_config.yaml` file is required, for example:

```yaml
# SVG Container Tags
"<|START_OF_SVG|>": "Marks the beginning of an SVG document"
"<|END_OF_SVG|>": "Marks the end of an SVG document"

# SVG Group Tags
"<|start_of_g|>": "Begins a group element in SVG for organizing related shapes"
"<|end_of_g|>": "Ends a group element"
```

In this file, you need to define both the special tokens and their corresponding descriptions.

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

`new_special_tokens_config` specifies the path to the `tokens_config.yaml` file, while `init_special_tokens` configures the method for initializing the embeddings of the special tokens. The `init_special_tokens` option can be either `desc_init` or `desc_init_w_noise`. Using an initialization method with token descriptions allows the tokenizer to initialize the token embeddings based on their descriptions.

**Note:** Loading special tokens via a file takes higher priority than specifying them directly in the configuration file.

### 4.3 Add via the Visual Interface

![image-20251217151544496](https://github.com/user-attachments/assets/5679a55d-f694-415c-8f51-0c0f9b2dbd25)

Simply add the content that would normally go into the YAML file under the `Extra arguments` section.