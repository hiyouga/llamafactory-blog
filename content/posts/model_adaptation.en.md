---
date: '2025-12-12T21:13:00+08:00'
draft: false
title: 'Adapt a new model on LLaMA-Factory'
---

## 1 Overview of Model Adaptation

LLaMA-Factory offers a complete framework for model pre-training, fine-tuning, and inference. If it is necessary to adapt a new model, only a small amount of code needs to be modified to integrate the model into LLaMA-Factory.

First, the file `LLaMA-Factory/src/llamafactory/extras/constants.py` defines the supported model groups and their corresponding templates. A template is a “format specifier” used when constructing the input prompt for the large model. It defines the dialogue format, field structure, role order, and the format for tool calls. For example:

```python
register_model_group(
    models={
        "Vicuna-v1.5-7B-Chat": {
            DownloadSource.DEFAULT: "lmsys/vicuna-7b-v1.5",
            DownloadSource.MODELSCOPE: "Xorbits/vicuna-7b-v1.5",
        },
        "Vicuna-v1.5-13B-Chat": {
            DownloadSource.DEFAULT: "lmsys/vicuna-13b-v1.5",
            DownloadSource.MODELSCOPE: "Xorbits/vicuna-13b-v1.5",
        },
    },
    template="vicuna",
```

The `template` must also be registered in `LLaMA-Factory/src/llamafactory/data/template.py`. **The template name can be chosen quite freely, as long as the name used in `constants.py` matches the template’s registered name.** The template is defined as follows:

```python
register_template(
    name="ministral3",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
)
```

`register_template` completes the registration of the template. The template above creates formatters for elements such as `user` and `system`. **The parameters of these formatters must refer to the model’s `chat_template.jinja` file**, from which the slot parameters are obtained. The template also creates an `mm_plugin` plugin, which is used to adapt inputs for multimodal models, such as audio and video. `get_mm_plugin` retrieves the multimodal plugins registered in `LLaMA-Factory/src/llamafactory/data/mm_plugin.py`.

**If the model is a multimodal model, you also need to register a `CompositeModel` object.** In `LLaMA-Factory/src/llamafactory/model/model_utils/visual.py`, this file must register a `CompositeModel` object to identify the vision module and language module. For example:

```python
_register_composite_model(
    model_type="qwen2_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)

_register_composite_model(
    model_type="mistral3",
    projector_key="model.multi_modal_projector",
)
```

The value of `model_type` must match the `model_type` field in the model’s configuration file, for example the `model_type` inside the [Ministral-3-3B-Instruct-2512/config.json](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/blob/main/config.json).

After completing the above steps, the model is basically adapted. However, you still need to verify whether training and inference function correctly. Below is an example of the specific model adaptation process using Ministral-3-3B-Instruct-2512.

## 2 Adapting Ministral-3-3B-Instruct-2512

Mistral 3 is a new generation of open-source AI model series released by Mistral AI (December 2025), including the smaller Ministral 3 models (3B, 8B, 14B parameters) and the larger Mistral Large 3 (675B total parameters, 41B active parameters). **The models support multimodality (text and images) and multilingual capabilities**, offering high performance and strong cost-effectiveness.

Mistral 3, combined with optimizations from partners such as NVIDIA, can run efficiently on various hardware platforms. It is suitable for scenarios such as edge computing and enterprise-level deployments, providing developers with powerful tools for building and deploying AI applications.

### 2.1 Model Registration

```python
register_model_group(
    models={
        "Ministral-3-3B-Instruct-2512": {
            DownloadSource.DEFAULT: "mistralai/Ministral-3-3B-Instruct-2512",
            DownloadSource.MODELSCOPE: "mistralai/Ministral-3-3B-Instruct-2512",
        },
        "Ministral-3-8B-Instruct-2512": {
            DownloadSource.DEFAULT: "mistralai/Ministral-3-8B-Instruct-2512",
            DownloadSource.MODELSCOPE: "mistralai/Ministral-3-8B-Instruct-2512",
        },
        "Ministral-3-14B-Instruct-2512": {
            DownloadSource.DEFAULT: "mistralai/Ministral-3-14B-Instruct-2512",
            DownloadSource.MODELSCOPE: "mistralai/Ministral-3-14B-Instruct-2512",
        },
    },
    template="ministral3",
    multimodal=True,
)
```

`DownloadSource.DEFAULT` is the download path for Hugging Face, and it needs to be copied correctly.

![image-20251211175942561](https://github.com/user-attachments/assets/eb0d2d8e-def8-4881-94b0-3a1a2be5b4b8)

`DownloadSource.MODELSCOPE` is the download path for ModelScope, and you must also ensure it is copied correctly.

![image-20251211180144183](https://github.com/user-attachments/assets/1a422453-f5a4-4128-8803-47fad20c43c0)

**The addresses for Hugging Face and ModelScope may differ, so please take care to distinguish them.**

The `template` is named `ministral3`, and since this model is multimodal, the parameter `multimodal` is set to `True`.

### 2.2 模版注册

```python
register_template(
    name="ministral3",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["[SYSTEM_PROMPT]{{content}}[/SYSTEM_PROMPT]"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
)
```

The `chat_template.jinja` of **[ministral3](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/blob/main/chat_template.jinja)** defines the model’s structure.

- format_user

```jinja2
{#- User messages supports text content or text and image chunks. #}
    {%- if message['role'] == 'user' %}
        {%- if message['content'] is string %}
            {{- '[INST]' + message['content'] + '[/INST]' }}
        {%- elif message['content'] | length > 0 %}
            {{- '[INST]' }}
            {%- if message['content'] | length == 2 %}
                {%- set blocks = message['content'] | sort(attribute='type') %}
            {%- else %}
                {%- set blocks = message['content'] %}
            {%- endif %}
            {%- for block in blocks %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- elif block['type'] in ['image', 'image_url'] %}
                    {{- '[IMG]' }}
                {%- else %}
                    {{- raise_exception('Only text, image and image_url chunks are supported in user message content.') }}
                {%- endif %}
            {%- endfor %}
            {{- '[/INST]' }}
        {%- else %}
            {{- raise_exception('User message must have a string or a list of chunks in content') }}
        {%- endif %}
```

According to the template, the template slot for user input is `"[INST]{{content}}[/INST]"`.

- format_system

```jinja2
{#- Handle system prompt if it exists. #}
{#- System prompt supports text content or text chunks. #}
{%- if messages[0]['role'] == 'system' %}
    {{- '[SYSTEM_PROMPT]' -}}
    {%- if messages[0]['content'] is string %}
        {{- messages[0]['content'] -}}
    {%- else %}        
        {%- for block in messages[0]['content'] %}
            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- else %}
                {{- raise_exception('Only text chunks are supported in system message contents.') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '[/SYSTEM_PROMPT]' -}}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
    {%- if default_system_message != '' %}
        {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}
    {%- endif %}
{%- endif %}
```

According to the template, the template slot for system input is `"[SYSTEM_PROMPT]{{content}}[/SYSTEM_PROMPT]"`.

- format_function

```jinja2
{%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}
    {%- for tool in message['tool_calls'] %}
        {%- set arguments = tool['function']['arguments'] %}
        {%- if arguments is not string %}
            {%- set arguments = arguments|tojson|safe %}
        {%- elif arguments == '' %}
            {%- set arguments = '{}' %}
        {%- endif %}
        {{- '[TOOL_CALLS]' + tool['function']['name'] + '[ARGS]' + arguments }}
    {%- endfor %}
{%- endif %}

{#- End of sequence token for each assistant messages. #}
{{- eos_token }}
```

According to the template, the formatting tool for `TOOL_CALLS` is:

```python
format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
```

- format_observation

```jinja2
{#- Tool messages only supports text content. #}
{%- elif message['role'] == 'tool' %}
    {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}
```

According to the template, the formatting tool for `TOOL_RESULTS` is:

```python
format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
```

- mm_plugin

Ministral3 can use the pixtral plugin, and the `image_token` is `[IMG]`.

```jinja2
{%- for block in blocks %}
    {%- if block['type'] == 'text' %}
        {{- block['text'] }}
    {%- elif block['type'] in ['image', 'image_url'] %}
        {{- '[IMG]' }}
    {%- else %}
        {{- raise_exception('Only text, image and image_url chunks are supported in user message content.') }}
    {%- endif %}
{%- endfor %}
```

### 2.3 Multimodal Registration

Ministral-3-3B-Instruct-2512 is a vision–language model, so you need to register the structural information of a multimodal model. These attributes must be set according to the vision and language module properties of the model.

```python
model_type: str,
projector_key: Optional[str] = None,
vision_model_keys: Optional[list[str]] = None,
language_model_keys: Optional[list[str]] = None,
lora_conflict_keys: Optional[list[str]] = None,
```

The multimodal information of Ministral3 is:

```python
_register_composite_model(
    model_type="mistral3",
    projector_key="model.multi_modal_projector",
)
```

### 2.4 Transformers Version

You also need to check the model’s `config.json` file to see which version of Transformers the model uses, ensuring that the model can load and run correctly.

```json
"transformers_version": "5.0.0.dev0",
```

### 2.5 细节问题

![image-20251212101021049](https://github.com/user-attachments/assets/1294d4df-7375-4cff-a7c1-c8f45e42c9ba)

Ministral-3-3B-Instruct-2512 is a quantized model. Transformers does not support training fp8 models, so it must be dequantized to a higher-precision model for training. Therefore, the model needs to be loaded in its dequantized form.

The `load_model` function in `LLaMA-Factory/src/llamafactory/model/loader.py` handles the model loading.

```python
def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))
```

The `patch_config` function handles some model parameter configurations and is located in `LLaMA-Factory/src/llamafactory/model/patcher.py`.

```python
def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)
```

The `configure_quantization` function configures the quantization parameters. You need to add the dequantization configuration for fp8 models under `if getattr(config, "quantization_config", None):`.

```python
def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    init_kwargs: dict[str, Any],
) -> None:
    r"""Priority: PTQ-quantized (train/infer) > AutoGPTQ (export) > On-the-fly quantization (train/infer)."""
    if getattr(config, "quantization_config", None):  # ptq
        if model_args.quantization_bit is not None:
            logger.warning_rank0("`quantization_bit` will not affect on the PTQ-quantized models.")

        quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        if quant_method != QuantizationMethod.MXFP4 and (is_deepspeed_zero3_enabled() or is_fsdp_enabled()):
            # mxfp4 will dequant the model weights
            raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")

        if quant_method == QuantizationMethod.GPTQ:
            check_version("gptqmodel>=2.0.0", mandatory=True)
            quantization_config.pop("disable_exllama", None)  # remove deprecated args
            quantization_config["use_exllama"] = False  # disable exllama

        if quant_method == QuantizationMethod.AWQ:
            check_version("autoawq", mandatory=True)

        if quant_method == QuantizationMethod.AQLM:
            check_version("aqlm>=1.1.0", mandatory=True)
            quantization_config["bits"] = 2
		
        # Add QuantizationMethod.FP8 dequantize
        if quant_method == QuantizationMethod.FP8 and is_trainable:
            quant_config = FineGrainedFP8Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
    ......
```

## 3 Model Testing

---

```bash
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 llamafactory-cli webui
```

Chat:

![image-20251212111231479](https://github.com/user-attachments/assets/09e47d2f-96bf-4579-a347-1638d71edba0)

Fine-Tuning:

![image-20251212112041712](https://github.com/user-attachments/assets/30c761c2-53e2-41af-96b2-77feb8e1263f)

![image-20251212112116132](https://github.com/user-attachments/assets/3945192e-e658-4ed3-be9d-7ba5511a91f4)