## 1 模型适配概述

LLaMA-Factory 提供了一整套模型预训练，微调，推理的框架，如果需要适配新模型，只需要修改少量代码就可以让模型融入 LLaMA-Factory 当中。

首先 `LLaMA-Factory/src/llamafactory/extras/constants.py` 文件定义了支持的模型组及其对应的 template。template 是在构建输入给大模型（prompt）时，用来规定对话格式、字段结构、角色顺序、工具调用格式的“格式规范器”。例如

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

`template` 也需要在 `LLaMA-Factory/src/llamafactory/data/template.py` 完成相应的注册，**模版的命名可以比较自由，只要保证 `constants.py` 使用的是对应的模版名即可**，模版定义如下：

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

`register_template` 完成模版的注册，上面这个模版完成了 `user`, `system` 等的格式化器的创建，这些**格式化器的入参需要参考这个模型的 chat_template.jinja 文件**，从 chat_template.jinja 文件获取到 slots 参数。上述模版还创建了 `mm_plugin` 插件，这个参数用于适配多模态模型的输入，例如音频，视频，`get_mm_plugin` 得到 `LLaMA-Factory/src/llamafactory/data/mm_plugin.py` 文件下注册的多模态插件。

**如果模型是一个多模态模型，还需要注册 `CompositeModel` 对象**，在 `LLaMA-Factory/src/llamafactory/model/model_utils/visual.py` 下，这个文件内部需要注册 `CompositeModel` 对象用于识别视觉模块和语言模块。例如：

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

其中 `model_type` 必须要和模型配置文件的 `model_type` 值一样，例如 [Ministral-3-3B-Instruct-2512/config.json](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/blob/main/config.json) 内部的 `model_type`。

完成了上述适配模型基本上适配好了，但是还需要验证训练，推理是否正确。下面以 Ministral-3-3B-Instruct-2512 为例介绍具体的模型适配流程。

## 2 Ministral-3-3B-Instruct-2512适配

Mistral 3 是 Mistral AI 推出的新一代开源 AI 模型系列（2025年12月），包括小型的 Ministral 3（3B、8B、14B 参数）和大型的 Mistral Large 3（675B 总参数，41B 激活参数）。**模型支持多模态（文本和图像）与多语言功能**，具有高性能和高性价比。Mistral 3 结合 NVIDIA 等合作伙伴的优化技术，可在多种硬件上高效运行，适用边缘计算、企业级部署等多种场景，为开发者提供强大的工具构建和部署 AI 应用。

### 2.1 模型注册

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

`DownloadSource.DEFAULT` 是 Hugging Face 的下载路径，需要正确复制

![image-20251211175942561](https://github.com/user-attachments/assets/eb0d2d8e-def8-4881-94b0-3a1a2be5b4b8)

`DownloadSource.MODELSCOPE` 是 ModelScope 的下载路径，也要保证正确复制

![image-20251211180144183](https://github.com/user-attachments/assets/1a422453-f5a4-4128-8803-47fad20c43c0)

**Hugging Face 和 ModelScope 的地址可能会不一样，请注意甄别。**

`template` 命名为 `ministral3`，这个模型是多模态参数，设置 multimodal 为 True。

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

ministral3 的 [chat_template.jinja](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/blob/main/chat_template.jinja) 定义了模型

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

根据模版情况，用户输入的模版槽是 `"[INST]{{content}}[/INST]"`。

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

根据模版情况，系统输入的模版槽是 `"[SYSTEM_PROMPT]{{content}}[/SYSTEM_PROMPT]"`。

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

根据模版情况，TOOL_CALLS 的格式工具为

```python
format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
```

- format_observation

```jinja2
{#- Tool messages only supports text content. #}
{%- elif message['role'] == 'tool' %}
    {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}
```

根据模版情况，TOOL_RESULTS 的格式工具为

```python
format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
```

- mm_plugin

Ministral3 可以使用 pixtral 的 plugin，然后 image_token 是 `[IMG]`。

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

### 2.3 多模态注册

Ministral-3-3B-Instruct-2512 是一个视觉语言模型，需要注册一个多模态模型的结构信息，需要根据模型的视觉语言模块属性去设置这些属性。

```python
model_type: str,
projector_key: Optional[str] = None,
vision_model_keys: Optional[list[str]] = None,
language_model_keys: Optional[list[str]] = None,
lora_conflict_keys: Optional[list[str]] = None,
```

Ministral3 的多模态信息为

```python
_register_composite_model(
    model_type="mistral3",
    projector_key="model.multi_modal_projector",
)
```

### 2.4 Transformers 版本

同样需要使用模型的 config.json 文件查看该模型使用的 Transformers 的版本，保证模型能够正常加载运行

```json
"transformers_version": "5.0.0.dev0",
```

### 2.5 细节问题

![image-20251212101021049](https://github.com/user-attachments/assets/1294d4df-7375-4cff-a7c1-c8f45e42c9ba)

Ministral-3-3B-Instruct-2512 是一个量化模型，Transformers 不支持 fp8 模型的训练，需要将其反量化为更高精度的模型才能训练。因此在加载模型的时候就需要以反量化的形式加载。

在 `LLaMA-Factory/src/llamafactory/model/loader.py` 文件下的 `load_model` 会完成模型的加载。

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

其中 `patch_config` 完成一些模型参数的配置，在 `LLaMA-Factory/src/llamafactory/model/patcher.py` 下

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

其中 `configure_quantization` 会配置量化参数，需要在 `if getattr(config, "quantization_config", None):` 下面添加 fp8 模型的反量化配置参数。

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
		
        # 添加 QuantizationMethod.FP8 的分支反量化参数
        if quant_method == QuantizationMethod.FP8 and is_trainable:
            quant_config = FineGrainedFP8Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
    ......
```

## 3 模型测试

---

```bash
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 llamafactory-cli webui
```

Chat:

![image-20251212111231479](https://github.com/user-attachments/assets/09e47d2f-96bf-4579-a347-1638d71edba0)

Fine-Tuning:

![image-20251212112041712](https://github.com/user-attachments/assets/30c761c2-53e2-41af-96b2-77feb8e1263f)

![image-20251212112116132](https://github.com/user-attachments/assets/3945192e-e658-4ed3-be9d-7ba5511a91f4)