import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk, Dataset
from torch.distributions.gamma import Gamma
from tqdm import tqdm
import random
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    GenerationMixin,
    Trainer, 
    TrainingArguments
)
import pynvml
from rich.console import Console
import subprocess   

import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache
from transformers import set_seed
from transformers import Trainer, TrainerCallback
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from utils.prompter import Prompter
from transformers.utils import logging

import torch
import torch.nn as nn
import os
import gc
import copy
import json
import wandb
from typing import Callable, List, Optional, Tuple, Union, Any
import math

# local import
from configuration_split import SplitConfig
from split_model import SplitModel, SplitModelForCausalLM
from make_prompt import make_chat_prompt, get_code_completion
from custom_callback import KLStepCallback, KLMetricsCallback, MemoryCleanupCallback

logger = logging.get_logger(__name__)

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import nn
from utils.prompter import Prompter
import sys
import psutil
import time
import os
import sys

wandb.init(project="split-model-with-nvib", name="split-model-with-nvib")

instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def update_model(model, tokenizer):
    model.to(device="cuda" if torch.cuda.is_available() else "cpu",dtype=torch.bfloat16)  # Only convert device, dtype is already correct
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    print(model.device)
    total_param = 0
    trainable_param = 0
    for param in model.parameters(): 
        param.requires_grad = False
        total_param += param.numel()
    for param in model.nvib_transformer_adapter1.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.nvib_transformer_adapter2.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.lm_head.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.layers[:model.config.enc_num_layers].parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.layers[-model.config.dec_num_layers:].parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.norm.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()

    model.model.embed_tokens.weight.requires_grad_(True)

    print(f'Total Parameters: {total_param:,}')
    print(f'Trainable Parameters: {trainable_param:,}')
    print(f'Non-trainable Parameters: {total_param - trainable_param:,}')
    
    # Print percentage of trainable parameters
    print(f'Percentage of trainable parameters: {100 * trainable_param / total_param:.2f}%')


def train(
    # model/data params
    base_model_path: str = f"meta-llama/Llama-3.1-8B-Instruct",  # the only required argument
    middle_model_path: str = f"Qwen/Qwen2.5-Coder-7B-Instruct", 
    # data_path: str = "iamtarun/python_code_instructions_18k_alpaca",
    data_path: str = "/project/phan/codellama/datasets/PGCodeTraining100k",
    # data_path: str = f"../datasets/PGCodeTraining68k.jsonl",
    output_dir: str = f"./temp-with-skip-connection-llama-Qwen-100k-highreg/",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 4000,
    val_set_size: int = 500,
    # lora hyperparams
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
    # resume_from_checkpoint: str = "/mmfs1/project/phan/tqn/Adapter/LLM-SplitModel/temp-with-merge-clean-7b/checkpoint-40228",  # either training checkpoint or final adapter
    resume_from_checkpoint: str = False,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
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
    gradient_accumulation_steps = 16
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
    )
    model = SplitModelForCausalLM(config=config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
    tokenizer.padding_side = "left" 
    # tokenizer = AutoTokenizer.from_pretrained(f"../dictionary")
    update_model(model, tokenizer)
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result

    template = "    \"\"\"{}\"\"\"\n"
    def generate_and_tokenize_prompt_auto_completion(data_point):
        if data_point["input"] != "":
            system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            
        temp_prompt =  data_point['output'].split(":")[0] + ":\n"+template.format(data_point['instruction'])
        
        # messages = make_chat_prompt(full_prompt[len(system)+2:],instruction_prefix, response_prefix, tokenizer)
        output = data_point["output"]
        temp_tokenizer = tokenizer(
                            output,
                            truncation=True,
                            max_length=cutoff_len,
                            padding=False,
                            return_tensors=None,
                         )
        if len(temp_tokenizer['input_ids']) == 1:
            full_prompt = prompter.generate_prompt(
                            data_point["instruction"],
                            data_point["input"],
                            "",
                        )
            messages = make_chat_prompt(full_prompt[len(system)+2:],instruction_prefix, response_prefix, tokenizer)+output
        
        else:
            random_numbers = sorted(random.sample(range(0, len(temp_tokenizer['input_ids'])), k=2))
            #print(random_numbers)
            if random_numbers[0] == 0:
                #prefix = temp_prompt + tokenizer.batch_decode([temp_tokenizer['input_ids'][random_numbers[0]]], skip_special_tokens=False)[0]
                prefix = tokenizer.batch_decode([temp_tokenizer['input_ids'][random_numbers[0]]], skip_special_tokens=False)[0]
            else:
                #prefix = temp_prompt + tokenizer.batch_decode([temp_tokenizer['input_ids'][:random_numbers[0]]], skip_special_tokens=False)[0]
                prefix = tokenizer.batch_decode([temp_tokenizer['input_ids'][:random_numbers[0]]], skip_special_tokens=False)[0]
            
            if random_numbers[1] == len(temp_tokenizer['input_ids']):
                suffix = "\"\"\""
            else:
                suffix = tokenizer.batch_decode([temp_tokenizer['input_ids'][random_numbers[1]:]], skip_special_tokens=False)[0]
        
            temp_prompt = get_code_completion(prefix, suffix)
            #print(temp_prompt)
            temp_prompt = prompter.generate_prompt(
                            temp_prompt,
                            data_point["input"],
                            "",
                        )
            
            messages = make_chat_prompt(temp_prompt,instruction_prefix, response_prefix, tokenizer)
            
            
            middle = tokenizer.batch_decode([temp_tokenizer['input_ids'][random_numbers[0]:random_numbers[1]]], skip_special_tokens=False)[0]
            #print(prefix+middle+suffix)
            messages += middle
            #return messages
            # exit()
        tokenized_full_prompt = tokenize(messages)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
    
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt


    #template = "{}\n```"
    
    def generate_and_tokenize_prompt(data_point):
        if data_point["input"] != "":
            system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            "",
        )
        
        messages = make_chat_prompt(full_prompt[len(system)+2:],instruction_prefix, response_prefix, tokenizer)
        
        output = data_point["output"]
        messages += output
        # print(messages)
        # exit()
        
        tokenized_full_prompt = tokenize(messages)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
    
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)
    # data['train'] = Dataset.from_dict(data['train'][0])
    
    #Loading Disk Datasets
    
    # data = load_from_disk(data_path)
    
    #   # Be more transparent about the % of trainable params.
    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     # train_data_generation = (
    #     #     train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     # )
    #     # train_data_auto_completion = (
    #     #     train_val["train"].shuffle().map(generate_and_tokenize_prompt_auto_completion)
    #     # )
    #     # train_data = concatenate_datasets([train_data_generation, train_data_auto_completion]).shuffle()


    #     # val_data_generation = (
    #     #     train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     # )
    #     # val_data_auto_completion = (
    #     #     train_val["test"].shuffle().map(generate_and_tokenize_prompt_auto_completion)
    #     # )
    #     # val_data = concatenate_datasets([val_data_generation, val_data_auto_completion]).shuffle()

    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
        
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
        
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    # Save train and validation data to JSON files
    # train_data.to_json("train_data_llama_100k.json")
    # val_data.to_json("val_data_llama_100k.json")

    #########################################################
    # reuse old dataset
    train_data = load_dataset('json', data_files="./data/train_data_llama_100k.json")["train"]
    val_data = load_dataset('json', data_files="./data/val_data_llama_100k.json")["train"]
    #########################################################
    
    if resume_from_checkpoint:
        train_data = load_dataset('json', data_files="./data/train_data_llama.json")["train"]
        val_data = load_dataset('json', data_files="./data/val_data_llama.json")["train"]
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            #model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    
    trainer = transformers.Trainer(
        model=model,  
        tokenizer = tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=750,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,   # Use bf16 for mixed precision with bfloat16 models
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps= 100 if val_set_size > 0 else None,
            save_steps=300,
            output_dir=output_dir,
            save_total_limit=1,
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
    
    model.save_pretrained(output_dir, safe_serialization=False)
    
    
   
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)

