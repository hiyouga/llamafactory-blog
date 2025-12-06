---
date: '2025-12-05T21:13:00+08:00'
draft: false
title: 'LLaMA Factory 项目代码指南'
---

## 1 LLaMa-Factory 项目简介

[LLaMA-Factory](https://github.com/tangefly/LLaMA-Factory/tree/main) 是一个面向大语言模型（LLM）的高效训练与微调框架，专为简化 LLaMA 系列以及各类开源大模型的训练流程而设计。它以“开箱即用、灵活高效”为核心理念，提供从数据准备、参数高效微调（PEFT）、训练配置管理到模型部署的一站式解决方案。

LLaMA-Factory 支持多种主流模型架构（如 LLaMA、Qwen、Gemma、Mistral 等），并集成了 LoRA、QLoRA、AdaLoRA、Prompt Tuning 等多种轻量化训练技术，使开发者能够以极低成本在单卡或多卡环境下完成高质量模型微调。

该框架提供直观易用的命令行工具与 Web UI，适配从科研实验到生产级应用的多场景需求。通过结构化的配置体系、完善的训练监控以及可扩展的数据加载管线，LLaMA-Factory 让大模型训练变得更加透明、可控且易于维护。

## 2 项目目录结构

```bash
LLaMA-Factory
├── assets                      # 项目静态资源（图标、示例图片、赞助商信息等）
│   ├── sponsors                # 赞助商Logo与展示资源
│   └── thirdparty              # 第三方依赖或引用资源
│
├── data                        # Demo 数据与示例数据集
│   └── mllm_demo_data          # 多模态LLM（MLLM）演示数据
│
├── docker                      # Docker 环境配置（CUDA/NPU/ROCm 等）
│   ├── docker-cuda             # NVIDIA GPU 环境 Dockerfile
│   ├── docker-npu              # 华为 Ascend NPU 训练环境
│   └── docker-rocm             # AMD ROCm 训练环境
│
├── examples                    # 训练、推理、工具使用的完整示例脚本
│   ├── accelerate              # accelerate 分布式训练示例
│   ├── ascend                  # 华为 Ascend NPU 示例
│   ├── deepspeed               # DeepSpeed 训练配置与示例
│   ├── extras                  # 各类高级特性示例（如 fp8、Galore、LoRA+ 等）
│   │   ├── adam_mini           # Adam mini 参数优化示例
│   │   ├── apollo              # Apollo 优化器示例
│   │   ├── badam               # BAdam 优化器示例
│   │   ├── dft                 # DFT 训练示例
│   │   ├── fp8                 # FP8 训练示例
│   │   ├── fsdp_qlora          # FSDP + QLoRA 示例
│   │   ├── galore              # GaLore低秩优化示例
│   │   ├── llama_pro           # LLaMA-Pro 示例
│   │   ├── loraplus            # LoRA+ 示例
│   │   ├── mod                 # MOD 技术示例
│   │   ├── multi_tokens        # 多Token生成实验示例
│   │   ├── muon                # Muon 优化器示例
│   │   ├── nlg_eval            # 文本生成质量评测示例
│   │   ├── oft                 # OFT 微调示例
│   │   ├── pissa               # PiSSA 权重重参数方法示例
│   │   └── qoft                # QOFT 示例
│   ├── inference               # 推理脚本示例（Chat、工具调用等）
│   ├── kt_optimize_rules       # KTO 规则示例（奖励建模优化）
│   ├── megatron                # Megatron-LM 适配示例
│   ├── merge_lora              # LoRA 权重合并示例
│   ├── train_full              # 全参数训练示例
│   ├── train_lora              # LoRA 微调示例
│   └── train_qlora             # QLoRA 微调示例
│
├── scripts                     # 辅助脚本（转换、统计、API示例）
│   ├── api_example             # HTTP/API 使用示例
│   ├── convert_ckpt            # 权重转换脚本（HF <-> 原始权重）
│   └── stat_utils              # 数据、token 统计工具
│
├── src
│   └── llamafactory            # 核心代码入口
│       ├── api                 # Web API / 服务端（OpenAI 接口兼容）
│       ├── chat                # ChatEngine、多轮对话、工具调用
│       ├── data                # 数据加载、处理、格式化、模板系统
│       │   ├── processor       # 数据预处理组件（指令/对话/多模态）
│       ├── eval                # 模型评测
│       ├── extras              # 辅助函数（如日志、常量，环境等）
│       ├── hparams             # 超参数解析（模型/数据/训练参数）
│       ├── model               # 模型加载、LoRA/QLoRA/PEFT、patch等
│       │   └── model_utils     # 模型结构、权重、分布式工具
│       ├── third_party         # 集成第三方模块（如 muon 优化器）
│       ├── train               # 训练核心模块（SFT/DPO/KTO/PPO/RM等）
│       │   ├── dpo             # DPO 训练逻辑
│       │   ├── ksft            # KSFT 训练逻辑
│       │   ├── kto             # KTO 训练逻辑
│       │   ├── mca             # MCA 对齐算法
│       │   ├── ppo             # PPO 强化学习训练
│       │   ├── pt              # 预训练流程
│       │   ├── rm              # Reward Model（奖励模型训练）
│       │   └── sft             # 指令监督微调
│       ├── v1                  # 老版本核心（LLaMA-Factory v1）
│       │   ├── config          # v1 配置系统
│       │   ├── core            # v1 核心训练与推理模块
│       │   ├── extras          # v1 扩展功能
│       │   ├── plugins         # 插件系统（模型/采样器/分布式等）
│       │   └── trainers        # 训练器（Trainer）体系
│       └── webui               # Web UI（Gradio 界面）
│           ├── components      # UI 组件
│
├── tests                       # 测试集（核心模块测试）
│   ├── data                    # 数据处理测试
│   ├── e2e                     # 端到端测试
│   ├── eval                    # 评测模块测试
│   ├── model                   # 模型加载/patch等测试
│   └── train                   # 训练流程测试
│
└── tests_v1                    # v1 版本的历史测试集
    ├── core                    # v1 核心模块测试
    └── plugins                 # v1 插件系统测试
```

## 3 命令行接口

在使用如下命令安装环境依赖时，setup.py 会注册命令行工具 

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,modelscope]"
```

在 setup.py 中，get_console_scripts 这个函数将定义控制台入口 `llamafactory-cli = llamafactory.cli:main`。

```python
def get_console_scripts() -> list[str]:
    console_scripts = ["llamafactory-cli = llamafactory.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("lmf = llamafactory.cli:main")

    return console_scripts
```

`llamafactory.cli:main` 是 `LLaMA-Factory/src/llamafactory/cli.py` 下的一个入口函数。

## 4 前后端交互逻辑

LLaMa-Factory 使用 [gradio](https://github.com/gradio-app/gradio?tab=readme-ov-file) 搭建项目 webui 和后端，gradio 是一个开源 Python 包，允许快速为机器学习模型、API 或任何任意 Python 函数**构建**一个演示或 Web 应用程序。然后，可以使用 Gradio 内置的共享功能，在几秒钟内**分享**演示或 Web 应用程序的链接。*无需 JavaScript、CSS 或 Web 托管经验！*

`LLaMa-Factory`  在 `/LLaMA-Factory/src/llamafactory/webui/interface.py` 文件下创建 UI 界面。`create_ui` 定义了 web 界面的设置。

```python
def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory ({hostname})", css=CSS) as demo:
        title = gr.HTML()
        subtitle = gr.HTML()
        if demo_mode:
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        engine.manager.add_elems("head", {"title": title, "subtitle": subtitle})
        engine.manager.add_elems("top", create_top())
        lang: gr.Dropdown = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("Train"):
            engine.manager.add_elems("train", create_train_tab(engine))

        with gr.Tab("Evaluate & Predict"):
            engine.manager.add_elems("eval", create_eval_tab(engine))

        with gr.Tab("Chat"):
            engine.manager.add_elems("infer", create_infer_tab(engine))

        if not demo_mode:
            with gr.Tab("Export"):
                engine.manager.add_elems("export", create_export_tab(engine))

        engine.manager.add_elems("footer", create_footer())
        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo
```

## 5 主要功能模块事件绑定

组件定义和事件绑定在这个 `LLaMA-Factory/src/llamafactory/webui/components` 文件夹下，以训练模块为例，简要讲解事件绑定方法。

![image-20251205145110556](https://github.com/user-attachments/assets/9f732658-cc50-421d-9592-d166b6500023)

`LLaMA-Factory/src/llamafactory/webui/components/train.py` 内部的 `create_train_tab` 函数创建了训练模块的组件内容。最核心的几个按钮是 Preview，Save arguments，Load arguments，Start，Abort。

![image-20251205145237940](https://github.com/user-attachments/assets/1ff62916-425c-4c1a-8b2a-ffa7d0ab6793)

这几个按钮分别绑定了对应的函数，只要点击按钮，就执行绑定的函数。

![image-20251202144017899](https://github.com/user-attachments/assets/9aef23bf-2fa5-4a72-b37c-0076b626e318)

## 6 训练按钮事件介绍

### 6.1 训练命令发起

下面以 `Start` 按钮为例，讲解项目核心代码的调用机制。“Start”按钮绑定了函数 `engine.runner.run_train` ，该函数在 `/LLaMA-Factory/src/llamafactory/webui/runner.py` 代码里面。

```python
def run_train(self, data):
    yield from self._launch(data, do_train=True)
```

`self._launch` 也在该文件内部，定义如下：

```python
def _launch(self, data: dict["Component", Any], do_train: bool) -> Generator[dict["Component", Any], None, None]:
        r"""Start the training process."""
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._initialize(data, do_train, from_preview=False)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            self.do_train, self.running_data = do_train, data
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)

            os.makedirs(args["output_dir"], exist_ok=True)
            save_args(os.path.join(args["output_dir"], LLAMABOARD_CONFIG), self._build_config_dict(data))

            env = deepcopy(os.environ)
            env["LLAMABOARD_ENABLED"] = "1"
            env["LLAMABOARD_WORKDIR"] = args["output_dir"]
            if args.get("deepspeed", None) is not None:
                env["FORCE_TORCHRUN"] = "1"

            # NOTE: DO NOT USE shell=True to avoid security risk
            self.trainer = Popen(["llamafactory-cli", "train", save_cmd(args)], env=env, stderr=PIPE, text=True)
            yield from self.monitor()
```

`_launch` 首先处理一些参数，提取出相应的模型参数后，调用 

```python
Popen(["llamafactory-cli", "train", save_cmd(args)], env=env, stderr=PIPE, text=True)
```

完成子进程的创建，开始训练。

### 6.2 llamafactory-cli train

llamafactory-cli 绑定的是 `/LLaMA-Factory/src/llamafactory/cli.py` 的 main 函数，

```python
def main():
    from .extras.misc import is_env_enabled

    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    else:
        from . import launcher

    launcher.launch()
```

接着调用 launcher.launch() 拉起服务，然后在 `/LLaMA-Factory/src/llamafactory/launcher.py` 下的 `launch()` 中进入 `train` 分支。

```python
elif command == "train":
    from .train.tuner import run_exp
    run_exp()
```

`run_exp()` 位于 `/LLaMA-Factory/src/train.py` 文件内。

```python
from llamafactory.train.tuner import run_exp

def main():
    run_exp()
```

而 `run_exp()` 来自于 `/LLaMA-Factory/src/llamafactory/train/tuner.py`

```python
def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})
```

`run_exp()` 检查是否使用 [ray](https://github.com/ray-project/ray) (一个开源的分布式机器学习加速引擎)，最终都会调用 `_training_function` 函数。

```python
def _training_function(config: dict[str, Any]) -> None:

    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    elif finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        if model_args.use_kt:
            from .ksft.workflow import run_sft as run_sft_kt

            run_sft_kt(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
        else:
            run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and ray.is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")
```

接着会根据不同的训练类型进入不同的入口函数。

### 6.3 以 run_sft 为例的代码逻辑

`/LLaMA-Factory/src/llamafactory/train` 目录下包含多个训练模式的调用代码，`_training_function` 会调用该目录下的代码进行训练。`run_sft` 调用的是 `/LLaMA-Factory/src/llamafactory/train/sft/workflow.py`。

```python
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
```

- load_tokenizer: 加载 tokenizer 和 processor
- get_template_and_fix_tokenizer: 根据配置选择合适的对话模板（Chat Template），并修正 Tokenizer 与模板之间的特殊 Token 匹配关系，使模型训练与推理时的对话格式正确。
- get_dataset: 获取数据集
- load_model: 加载模型权重

### 6.4 训练模型

加载好模型之后，调用 `/LLaMA-Factory/src/llamafactory/train/sft/trainer.py` 的训练函数训练模型

```python
# Initialize our Trainer
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    finetuning_args=finetuning_args,
    data_collator=data_collator,
    callbacks=callbacks,
    gen_kwargs=gen_kwargs,
    **dataset_module,
    **tokenizer_module,
    **metric_module,
)

# Training
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    if finetuning_args.include_effective_tokens_per_second:
        train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
            dataset_module["train_dataset"], train_result.metrics, stage="sft"
        )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += sum(
                [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
            )
        else:
            keys += ["eval_loss", "eval_accuracy"]

        plot_loss(training_args.output_dir, keys=keys)

if training_args.predict_with_generate:
    tokenizer.padding_side = "left"  # use left-padding in generation

# Evaluation
if training_args.do_eval:
    metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# Predict
if training_args.do_predict:
    logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
    predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
    trainer.log_metrics("predict", predict_results.metrics)
    trainer.save_metrics("predict", predict_results.metrics)
    trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

# Create model card
create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
```