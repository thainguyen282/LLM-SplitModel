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
from configuration_split import SplitConfig
from split_model import SplitModel, SplitModelForCausalLM
from make_prompt import make_chat_prompt, get_code_completion
from custom_callback import KLStepCallback, KLMetricsCallback, MemoryCleanupCallback, GenerateOnTrainExampleCallback
from custom_data_loader import LoadData
from update_trainable_parameters import update_trainable_parameters

wandb.init(project="split-model-with-nvib", name="split-model-with-nvib")

def train(
    # model/data params
    base_model_path: str = f"Qwen/Qwen2.5-Coder-7B-Instruct",
    middle_model_path: str = f"meta-llama/Llama-3.1-8B-Instruct",
    data_path: str = "/project/phan/codellama/datasets/PGCodeTraining100k",
    output_dir: str = f"./temp-with-Qwen-Llama-100k/",
    # training hyperparams
    batch_size: int = 1,
    num_epochs: int = 2,
    learning_rate: float = 3e-5,
    cutoff_len: int = 4000,
    val_set_size: int = 500,
    warmup_steps: int = 750,
    gradient_accumulation_steps: int = 16, 
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    resume_from_checkpoint: str = False,
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.

    #data preprocessing hyperparams
    using_raw_data = False
):
    prompter = Prompter(prompt_template_name)

    
    # model initialization
    AutoConfig.register("split", SplitConfig)
    AutoModel.register(SplitConfig, SplitModel)
    AutoModelForCausalLM.register(SplitConfig, SplitModelForCausalLM)
    config = SplitConfig(
        base_model_path=base_model_path, 
        middle_model_path=middle_model_path,
    )
    model = SplitModelForCausalLM(config=config)
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
    tokenizer.padding_side = "left" 

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    update_trainable_parameters(model, tokenizer)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     # model_args.model_name_or_path,
    #     # f"../../../tokenizerDP/Qwen2.5",
    #     f"/project/phan/codellama/tokenizerDP/Qwen2.5",
    #     # pad_token = '<|endoftext|>',
    #     # eos_token = '<|im_end|>', #<|endoftext|>
    #     # #cache_dir = "/mmfs1/project/phan/codellama/codellama/Qwen2.5-32B/",
    #     # #model_max_length = training_args.model_max_length,
    #     # #truncation = True,
    #     # use_fast=False,
    #     # padding_side = "right",
    #     # trust_remote_code = True
    # )

    # tokenizer.padding_side = "right" 

    # def update_model(eps, model, save_dir):
    #     device = model.device
    #     if eps != 0:
    #         file_name = f"./Qwen2.5_eps27.0_final.pt"
    #         # file_name = f"./original_new.pt"
    #         # file_name = f"CodeQwen_eps{eps}_new.pt"
    #         file_path = os.path.join(save_dir,file_name)
    #         new = torch.load(file_path)
    #         newEmbed = nn.Embedding(model.config.vocab_size, model.config.hidden_size, _weight=new.to(device))
    #         model.model.set_input_embeddings(newEmbed)

    # save_dir=f"/project/phan/codellama/Tensor/Qwen2.5-Coder"
    # update_model(27, model, save_dir)
    # model.config.bos_token_id = 72238
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.generation_config.bos_token_id = 72238
    # model.generation_config.eos_token_id = tokenizer.eos_token_id
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    # update_trainable_parameters(model, tokenizer)

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
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True, 
            logging_steps=20,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            max_grad_norm=1.0,
        ),
        # callbacks=[KLStepCallback(), KLMetricsCallback(), MemoryCleanupCallback(), GenerateOnTrainExampleCallback(tokenizer, model, train_data, every_n_steps=300)],
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

