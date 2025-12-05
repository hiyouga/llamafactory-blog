---
date: '2025-04-03T21:13:00+08:00'
draft: false
title: 'Code Guide for LLaMA Factory Project'
---

## 1 Introduction to the LLaMA-Factory Project

[LLaMA-Factory](https://github.com/tangefly/LLaMA-Factory/tree/main) is an efficient training and fine-tuning framework designed for large language models (LLMs). It aims to simplify the training workflow of the LLaMA family as well as various open-source large models. With the core philosophy of being “out-of-the-box, flexible, and efficient,” it provides an end-to-end solution covering data preparation, parameter-efficient fine-tuning (PEFT), training configuration management, and model deployment.

LLaMA-Factory supports multiple mainstream model architectures—such as LLaMA, Qwen, Gemma, and Mistral—and integrates lightweight training techniques including LoRA, QLoRA, AdaLoRA, and Prompt Tuning. These capabilities enable developers to fine-tune high-quality models at extremely low cost, whether in single-GPU or multi-GPU environments.

The framework offers intuitive command-line tools and a user-friendly web UI, making it suitable for a wide range of scenarios from research experiments to production-grade applications. Through its structured configuration system, comprehensive training monitoring, and extensible data loading pipeline, LLaMA-Factory makes large-model training more transparent, controllable, and maintainable.

## 2 Project Directory Structure

```bash
LLaMA-Factory
├── assets                      # Static assets of the project (icons, sample images, sponsor info, etc.)
│   ├── sponsors                # Sponsor logos and display resources
│   └── thirdparty              # Third-party dependencies or referenced resources
│
├── data                        # Demo data and sample datasets
│   └── mllm_demo_data          # Demo data for multimodal LLMs (MLLM)
│
├── docker                      # Docker environment configurations (CUDA/NPU/ROCm, etc.)
│   ├── docker-cuda             # Dockerfile for NVIDIA GPU environments
│   ├── docker-npu              # Huawei Ascend NPU training environment
│   └── docker-rocm             # AMD ROCm training environment
│
├── examples                    # Complete example scripts for training, inference, and tool usage
│   ├── accelerate              # accelerate distributed training examples
│   ├── ascend                  # Huawei Ascend NPU examples
│   ├── deepspeed               # DeepSpeed training configs and examples
│   ├── extras                  # Advanced feature examples (e.g., fp8, Galore, LoRA+, etc.)
│   │   ├── adam_mini           # Adam mini optimizer examples
│   │   ├── apollo              # Apollo optimizer examples
│   │   ├── badam               # BAdam optimizer examples
│   │   ├── dft                 # DFT training examples
│   │   ├── fp8                 # FP8 training examples
│   │   ├── fsdp_qlora          # FSDP + QLoRA examples
│   │   ├── galore              # GaLore low-rank optimization examples
│   │   ├── llama_pro           # LLaMA-Pro examples
│   │   ├── loraplus            # LoRA+ examples
│   │   ├── mod                 # MOD technique examples
│   │   ├── multi_tokens        # Multi-token generation experiments
│   │   ├── muon                # Muon optimizer examples
│   │   ├── nlg_eval            # Text generation evaluation examples
│   │   ├── oft                 # OFT fine-tuning examples
│   │   ├── pissa               # PiSSA re-parameterization examples
│   │   └── qoft                # QOFT examples
│   ├── inference               # Inference scripts (Chat, tool calling, etc.)
│   ├── kt_optimize_rules       # KTO rule examples (reward modeling optimization)
│   ├── megatron                # Megatron-LM adaptation examples
│   ├── merge_lora              # LoRA weight merging examples
│   ├── train_full              # Full-parameter training examples
│   ├── train_lora              # LoRA fine-tuning examples
│   └── train_qlora             # QLoRA fine-tuning examples
│
├── scripts                     # Utility scripts (conversion, statistics, API examples)
│   ├── api_example             # HTTP/API usage examples
│   ├── convert_ckpt            # Weight conversion scripts (HF <-> original weights)
│   └── stat_utils              # Data and token statistics tools
│
├── src
│   └── llamafactory            # Core code entry
│       ├── api                 # Web API / server (OpenAI-compatible interface)
│       ├── chat                # Chat engine, multi-turn dialogue, tool calling
│       ├── data                # Data loading, processing, formatting, and template system
│       │   ├── processor       # Data preprocessing components (instruction/dialogue/multimodal)
│       ├── eval                # Model evaluation
│       ├── extras              # Utilities (logging, constants, environment helpers, etc.)
│       ├── hparams             # Hyperparameter parsing (model/data/training)
│       ├── model               # Model loading, LoRA/QLoRA/PEFT, patches, etc.
│       │   └── model_utils     # Model structure, weights, and distributed utilities
│       ├── third_party         # Integrated third-party modules (e.g., muon optimizer)
│       ├── train               # Core training modules (SFT/DPO/KTO/PPO/RM, etc.)
│       │   ├── dpo             # DPO training logic
│       │   ├── ksft            # KSFT training logic
│       │   ├── kto             # KTO training logic
│       │   ├── mca             # MCA alignment algorithm
│       │   ├── ppo             # PPO reinforcement learning training
│       │   ├── pt              # Pretraining pipeline
│       │   ├── rm              # Reward model training
│       │   └── sft             # Instruction-based supervised fine-tuning
│       ├── v1                  # Legacy core (LLaMA-Factory v1)
│       │   ├── config          # v1 configuration system
│       │   ├── core            # v1 core training/inference modules
│       │   ├── extras          # v1 extensions
│       │   ├── plugins         # Plugin system (models/samplers/distributed, etc.)
│       │   └── trainers        # Trainer system
│       └── webui               # Web UI (Gradio interface)
│           ├── components      # UI components
│
├── tests                       # Test suite (core module tests)
│   ├── data                    # Data processing tests
│   ├── e2e                     # End-to-end tests
│   ├── eval                    # Evaluation module tests
│   ├── model                   # Model loading/patch tests
│   └── train                   # Training pipeline tests
│
└── tests_v1                    # Historical test suite for v1
    ├── core                    # v1 core tests
    └── plugins                 # v1 plugin system tests
```

## 3 Command-line interface

When installing the environment dependencies using the following command, setup.py will register the command-line tools.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,modelscope]"
```

In setup.py, the get_console_scripts function defines the console entry point `llamafactory-cli = llamafactory.cli:main`.

```python
def get_console_scripts() -> list[str]:
    console_scripts = ["llamafactory-cli = llamafactory.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("lmf = llamafactory.cli:main")

    return console_scripts
```

`llamafactory.cli:main` is an entry function located in `LLaMA-Factory/src/llamafactory/cli.py`.

## 4 Front-end and back-end interaction logic

LLaMA-Factory builds the project **web UI** and backend using [gradio](https://github.com/gradio-app/gradio?tab=readme-ov-file).
Gradio is an open-source Python package that allows you to quickly **build** a demo or web application for a machine learning model, API, or any arbitrary Python function.
You can then **share** the demo or web application link within seconds using Gradio’s built-in sharing features.
*No JavaScript, CSS, or web hosting experience required!*

`LLaMA-Factory` creates the UI interface in the file `/LLaMA-Factory/src/llamafactory/webui/interface.py`.
The `create_ui` function defines the configuration of the web interface.

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

## 5 Binding event handlers for the main functional modules

The component definitions and event bindings are located in the `LLaMA-Factory/src/llamafactory/webui/components` folder.
Taking the training module as an example, we briefly explain the method of event binding.

![image-20251205145110556](https://github.com/user-attachments/assets/9f732658-cc50-421d-9592-d166b6500023)

The `create_train_tab` function inside `LLaMA-Factory/src/llamafactory/webui/components/train.py` creates the component contents for the training module.
The key buttons include **Preview**, **Save arguments**, **Load arguments**, **Start**, and **Abort**.

![image-20251205145237940](https://github.com/user-attachments/assets/1ff62916-425c-4c1a-8b2a-ffa7d0ab6793)

Each of these buttons is bound to its corresponding function, and clicking a button triggers the execution of the function it is linked to.

![image-20251202144017899](https://github.com/user-attachments/assets/9aef23bf-2fa5-4a72-b37c-0076b626e318)

## 6 Introduction to the training button events

### 6.1 Launch the training command

Using the `Start` button as an example, we explain the invocation mechanism of the project’s core code.
The “Start” button is bound to the function `engine.runner.run_train`, which is located in the file `/LLaMA-Factory/src/llamafactory/webui/runner.py`.

```python
def run_train(self, data):
    yield from self._launch(data, do_train=True)
```

`self._launch` is also defined inside this file, and its definition is as follows:

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

`_launch` first processes several parameters and, after extracting the corresponding model arguments, calls…

```python
Popen(["llamafactory-cli", "train", save_cmd(args)], env=env, stderr=PIPE, text=True)
```

…completes the creation of the subprocess and starts the training.

### 6.2 llamafactory-cli train

`llamafactory-cli` is bound to the `main` function in `/LLaMA-Factory/src/llamafactory/cli.py`.

```python
def main():
    from .extras.misc import is_env_enabled

    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    else:
        from . import launcher

    launcher.launch()
```

It then calls `launcher.launch()` to start the service, and within the `launch()` function in `/LLaMA-Factory/src/llamafactory/launcher.py`, it enters the `train` branch.

```python
elif command == "train":
    from .train.tuner import run_exp
    run_exp()
```

`run_exp()` is located in the file `/LLaMA-Factory/src/train.py`.

```python
from llamafactory.train.tuner import run_exp

def main():
    run_exp()
```

And `run_exp()` comes from `/LLaMA-Factory/src/llamafactory/train/tuner.py`.

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

`run_exp()` checks whether [ray](https://github.com/ray-project/ray) is being used (an open-source distributed machine learning acceleration engine).
Regardless of the mode, it ultimately calls the `_training_function` function.

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

It then enters different entry functions depending on the training type.

### 6.3 The code logic using run_sft as an example

The `/LLaMA-Factory/src/llamafactory/train` directory contains the implementation code for multiple training modes, and `_training_function` invokes the code within this directory to perform training.
`run_sft` calls the module located at `/LLaMA-Factory/src/llamafactory/train/sft/workflow.py`.

```python
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
```

- **load_tokenizer**: Loads the tokenizer and processor.
- **get_template_and_fix_tokenizer**: Selects the appropriate chat template based on the configuration, and adjusts the tokenizer to correctly match the special tokens required by the template, ensuring accurate dialogue formatting during training and inference.
- **get_dataset**: Loads the dataset.
- **load_model**: Loads the model weights.


### 6.4 Training the model

After the model is loaded, the training function in `/LLaMA-Factory/src/llamafactory/train/sft/trainer.py` is called to train the model.

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
