import os
import torch
import fire
import wandb
import random
from typing import List
from torch import nn

from datasets import load_dataset
import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from utils.prompter import Prompter

# Local imports
from split_model.config import SplitConfig
from split_model.model import SplitModel, SplitModelForCausalLM
from utils.make_prompt import make_chat_prompt, get_code_completion
from utils.custom_callback import KLStepCallback, KLMetricsCallback, MemoryCleanupCallback
from utils.custom_data_loader import LoadData
from utils.update_trainable_parameters import update_trainable_parameters

wandb.init(project="split-model-with-nvib", name="split-model-with-nvib")

def train(
    # model/data params
    base_model_path: str = f"Qwen/Qwen2.5-Coder-7B-Instruct",
    middle_model_path: str = f"meta-llama/Llama-3.1-8B-Instruct",
    data_path: str = "iamtarun/python_code_instructions_18k_alpaca",
    output_dir: str = f"./saves/merge_model/",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 4000,
    val_set_size: int = 500,
    warmup_steps: int = 750,
    gradient_accumulation_steps: int = 16, 
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False, 
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",   
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.

    #data preprocessing hyperparams
    using_raw_data = True
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"base_model_path: {base_model_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model_path
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    
    # model initialization
    AutoConfig.register("split", SplitConfig)
    AutoModel.register(SplitConfig, SplitModel)
    AutoModelForCausalLM.register(SplitConfig, SplitModelForCausalLM)
    config = SplitConfig(
        base_model_path=base_model_path, 
        middle_model_path=middle_model_path,
    )
    model = SplitModelForCausalLM(config=config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.train()
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    update_trainable_parameters(model, tokenizer)

    #loading data
    loader = LoadData(data_path, tokenizer, prompter, cutoff_len, train_on_inputs, using_raw_data, val_set_size)
    train_data = loader.train_data
    val_data = loader.val_data
    
    trainer = transformers.Trainer(
        model=model,  
        tokenizer = tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,   # Use bf16 for mixed precision with bfloat16 models
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            metric_for_best_model="eval_loss",
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            # Add gradient clipping for numerical stability
            max_grad_norm=1.0,
        ),
        callbacks=[KLStepCallback(), KLMetricsCallback(), MemoryCleanupCallback()],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)    
   
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)

