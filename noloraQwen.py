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
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from transformers.utils import logging

# Local imports
from configuration_split import SplitConfig
from split_model import SplitModel, SplitModelForCausalLM
from make_prompt import make_chat_prompt, get_code_completion
from custom_callback import KLStepCallback, KLMetricsCallback, MemoryCleanupCallback
from custom_data_loader import LoadData
from update_trainable_parameters import update_trainable_parameters

wandb.init(project="split-model-with-nvib", name="split-model-with-nvib")

def train(
    # model/data params
    base_model_path: str = f"Qwen/Qwen2.5-Coder-7B-Instruct",
    middle_model_path: str = f"meta-llama/Llama-3.1-8B-Instruct",
    # data_path: str = "/project/phan/codellama/datasets/PGCodeTrainingCombined",
    data_path: str = "/project/phan/codellama/datasets/PGCodeTraining100k",
    # data_path: str = "iamtarun/python_code_instructions_18k_alpaca",
    output_dir: str = f"./temp-with-Qwen-Llama-eps27/",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 4000,
    val_set_size: int = 500,
    warmup_steps: int = 750,
    gradient_accumulation_steps: int = 16, 
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        'q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj',
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",   
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    # resume_from_checkpoint: str = "/project/phan/tqn/Adapter/LLM-SplitModel/temp-with-Qwen-Llama-final-2nd/checkpoint-3600",  # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.

    #data preprocessing hyperparams
    using_raw_data = False
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model_path: {base_model_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
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
   #tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        # model_args.model_name_or_path,
        # f"../../../tokenizerDP/Qwen2.5",
        f"/project/phan/codellama/tokenizerDP/Qwen2.5",
        # pad_token = '<|endoftext|>',
        # eos_token = '<|im_end|>', #<|endoftext|>
        # #cache_dir = "/mmfs1/project/phan/codellama/codellama/Qwen2.5-32B/",
        # #model_max_length = training_args.model_max_length,
        # #truncation = True,
        # use_fast=False,
        # padding_side = "right",
        # trust_remote_code = True
    )

    tokenizer.padding_side = "right" 

    def update_model(eps, model, save_dir):
        device = model.device
        if eps != 0:
            file_name = f"./Qwen2.5_eps27.0_final.pt"
            # file_name = f"./original_new.pt"
            # file_name = f"CodeQwen_eps{eps}_new.pt"
            file_path = os.path.join(save_dir,file_name)
            new = torch.load(file_path)
            newEmbed = nn.Embedding(model.config.vocab_size, model.config.hidden_size, _weight=new.to(device))
            model.model.set_input_embeddings(newEmbed)

    save_dir=f"/project/phan/codellama/Tensor/Qwen2.5-Coder"
    update_model(27, model, save_dir)
    model.config.bos_token_id = 72238
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = 72238
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
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

