---
date: '2025-12-17T14:13:17+08:00'
draft: true
title: 'Add New Special Tokens for Model Training'
---

## 1 Introduction

This paper uses the **Qwen2.5-3B-Instruct** model and applies **supervised fine-tuning (SFT)** to demonstrate how to add new **special tokens**. The experiment is run using the following command:

```bash
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train_lora/qwen2.5_lora_sft.yaml
```

The `qwen2.5_lora_sft.yaml` file needs to be configured in advance.

## 2 Overview of Dataset Loading

The data loading logic is implemented in `LLaMA-Factory/src/llamafactory/data/loader.py`. Within this file, the `get_dataset` function is responsible for loading the dataset and preprocessing the data using the tokenizer.


```python
def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

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

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)

```

- Data Loading

The following code completes data loading and converts the data format.

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

The loaded data is stored in `dataset`, and its format is converted as follows, for example:

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

- Data Preprocessing

```python
with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
    dataset = _get_preprocessed_dataset(
        dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
    )
```

**This code completes the conversion of data from `json` format into a structured format**, for example:

```
'_prompt': [{'role': 'user', 'content': 'Transform the following sentence using a synonym: The car sped quickly.'}]
```

Convert to

```
'<|im_start|>user\nTransform the following sentence using a synonym: The car sped quickly.<|im_end|>\n<|im_start|>assistant\n'
```

It also completes the tokenization process, with the function call flow shown below:

`_get_preprocessed_dataset` $\rightarrow$ `SupervisedDatasetProcessor.preprocess_dataset` $\rightarrow$ `SupervisedDatasetProcessor._encode_data_example` $\rightarrow$ `SupervisedDatasetProcessor.template.encode_multiturn` $\rightarrow$ `Template._encode`

- `LLaMA-Factory/src/llamafactory/data/template.py`

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

This function first performs format conversion and then uses the `tokenizer` to convert `elements` into token IDs.

## 3 **Add Special Tokens**

**Adding Special Tokens requires using the `add_special_tokens` interface of the `tokenizer`, as shown below:**

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

Therefore, to add **Special Tokens** in **LLaMA-Factory**, the required Special Tokens must be added to the **tokenizer**.

### 3.1 **Tokenizer Loading Method**

In `LLaMA-Factory/src/llamafactory/train/sft/workflow.py`, the tokenizer is loaded within the `run_sft` function.

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

The function call path is: `load_tokenizer` $\rightarrow$ `patch_tokenizer`.

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

It can be seen that if `model_args` contains the `add_special_tokens` parameter, then `add_special_tokens` will be loaded.

### 3.2 **model_args Loading Method**

Now that we understand how the tokenizer is loaded, the key question is how to load `model_args` and its internal `add_special_tokens`.

In `LLaMA-Factory/src/llamafactory/train/tuner.py`, the `_training_function` function reads the model arguments, data arguments, training arguments, and so on.

```python
def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    ......
```

The definition of `get_train_args` is as follows:

```python
def get_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False
    ......
```

The definition of `_parse_train_args` is as follows:

```python
def _parse_train_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)
```

The final parsing process is as follows:

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

`parser: "HfArgumentParser"` parses all arguments defined in `_TRAIN_ARGS` within `parser = HfArgumentParser(_TRAIN_ARGS)`, including `model_args`.

### 3.3 **Example of Adding Special Tokens**

To add special tokens, you only need to include the `add_special_tokens` parameter in the training configuration file, for example:

```yaml
### model
model_name_or_path: /home/xiaoxunpeng/workplace/Models/Qwen2.5-3B-Instruct
trust_remote_code: true
add_special_tokens: "[start],[end]"
...
```