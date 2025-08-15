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
    GenerationConfig,
)
import pynvml
from transformers import TrainerCallback
from rich.console import Console
import subprocess

import transformers
from nvib.denoising_attention import DenoisingMultiheadAttention
from nvib.nvib_layer import Nvib
from nvib_selfattention.nvib_sa_transformer_encoder import (
    NVIBTransformerEncoder,
    NVIBTransformerEncoderLayer,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache
from transformers import set_seed
from transformers import Trainer, TrainerCallback
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from utils.prompter import Prompter
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

import torch
import torch.nn as nn
import os
import gc
import copy
import json
import wandb
from typing import Callable, List, Optional, Tuple, Union, Any
import math
from configuration_split import SplitConfig
from support_split_model import init_weights, weighted_mean
from kl_annealing import kl_annealing
from make_prompt import make_chat_prompt

instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

class SplitModel(PreTrainedModel, GenerationMixin):
    config_class = SplitConfig
    def __init__(self, config):
        super().__init__(config)
    

class SplitModelForCausalLM(SplitModel):

    def __init__(self, config: SplitConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False
        
        ref_config = AutoConfig.from_pretrained(config.base_model_path)
        ref_config.tie_word_embeddings = False
        reference_model = AutoModelForCausalLM.from_config(
            ref_config,
            attn_implementation=config.attn_implementation, 
            torch_dtype=torch.bfloat16
        )
        
        # Initialize middle model from config if specified
        if config.middle_model_path is not None:
            middle_config = AutoConfig.from_pretrained(config.middle_model_path)
            self.middle_model = AutoModel.from_config(
                middle_config,
                attn_implementation=config.attn_implementation,
                torch_dtype=torch.bfloat16
            )
        else:
            self.middle_model = None
        
        self.model = reference_model.model
        self.lm_head = reference_model.lm_head

        # split model params
        self.enc_num_layers = config.enc_num_layers
        self.dec_num_layers = config.dec_num_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.is_nvib = config.is_nvib
        self.dropout = config.dropout
        self.weighted_kl = config.weighted_kl
        self.lambda_kld = config.lambda_kld
        self.lambda_klg = config.lambda_klg
        self.is_merge = config.is_merge
        
        # NVIB Transformer encoder layers 
        nvib_transformer_layer1 = NVIBTransformerEncoderLayer(
            in_dim=config.hidden_size,
            out_dim=config.compress_dim,
            nhead=config.nhead,
            dim_feedforward=config.compress_intermediate_size,
            dropout=config.dropout,
            activation="relu",
            kappa=config.kappa,
            delta=config.delta,
            batch_first=True,
            dtype=torch.bfloat16,  # Ensure NVIB layers are created with bfloat16
            norm_first=True
        )
        nvib_transformer_layer2 = NVIBTransformerEncoderLayer(
            in_dim=config.compress_dim,
            out_dim=config.hidden_size,
            nhead=config.nhead,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="relu",
            kappa=config.kappa,
            delta=config.delta,
            batch_first=True,
            dtype=torch.bfloat16,  # Ensure NVIB layers are created with bfloat16
            norm_first=True
        )
        encoder_norm1 = nn.LayerNorm(config.compress_dim, eps=1e-5, dtype=torch.bfloat16)  # Ensure LayerNorm is bfloat16
        encoder_norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5, dtype=torch.bfloat16)  # Ensure LayerNorm is bfloat16
        self.nvib_transformer_adapter1 = NVIBTransformerEncoder(
            encoder_layer=nvib_transformer_layer1, 
            num_layers=config.num_nvib_encoder_layers, 
            norm=encoder_norm1,
        )
        self.nvib_transformer_adapter2 = NVIBTransformerEncoder(
            encoder_layer=nvib_transformer_layer2, 
            num_layers=config.num_nvib_encoder_layers, 
            norm=encoder_norm2,
        )
        self.kl_annealing_scheduler = (
            kl_annealing(
                annealing_value_start=0,
                annealing_value_end=1,
                wait_before_warmup=200,
                end_of_warmup=600,
                type="linear",
            )
            if self.is_nvib else None
        )
        init_weights(self.nvib_transformer_adapter1)
        init_weights(self.nvib_transformer_adapter2)
        self.hidden_states = None
        self.kld = None
        self.klg = None
        del reference_model
        gc.collect()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = None # this if for flash attention 2 since it doesn't need causal_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # decoder layers
        next_decoder_cache = None

        kld = torch.tensor(0.0, device=hidden_states.device)
        klg = torch.tensor(0.0, device=hidden_states.device)
        # blue print
        layer_sequence = []
        for i in range(self.enc_num_layers):
            layer_sequence.append(('client', i, self.model.layers[i]))
        if self.is_nvib:    
            layer_sequence.append(('nvib1', None, self.nvib_transformer_adapter1))
        if self.is_merge:
            for i, middle_layer in enumerate(self.middle_model.layers):
                layer_sequence.append(('middle', i, middle_layer))
        for i in range(self.enc_num_layers, self.num_hidden_layers):
            if self.is_merge:
                if i < self.num_hidden_layers - self.dec_num_layers:
                    continue
            if i == self.num_hidden_layers - self.dec_num_layers and self.is_nvib:
                layer_sequence.append(('nvib2', None, self.nvib_transformer_adapter2))

            layer_sequence.append(('client', i, self.model.layers[i]))

        # start forward
        encoder_hidden_states = None
        for layer_type, layer_idx, layer in layer_sequence:
            if layer_type == 'nvib1':
                encoder_hidden_states = hidden_states
                
                hidden_states, _, kld, klg, _ = self.apply_nvib_adapter(
                    layer, hidden_states, attention_mask, self.kl_annealing_scheduler, 0, self.weighted_kl, self.lambda_kld, self.lambda_klg, kld, klg, self.training, "klg1"
                )
            elif layer_type == 'nvib2':
                hidden_states = self.middle_model.norm(hidden_states)
                
                hidden_states, _, kld, klg, _ = self.apply_nvib_adapter(
                    layer, hidden_states, attention_mask, self.kl_annealing_scheduler, 0, self.weighted_kl, self.lambda_kld, self.lambda_klg, kld, klg, self.training, "klg2"
                )
                hidden_states = hidden_states + encoder_hidden_states
            else:  # 'client' or 'middle' layers
                layer_output = layer(
                    hidden_states,
                    # attention_mask=causal_mask,
                    attention_mask=None, # this is for flashattention
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = layer_output[0]
        hidden_states = self.model.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)
            if self.lambda_klg != 0 or self.lambda_kld != 0: 
                loss = loss + kld + klg

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )

    def apply_nvib_adapter(
        self,
        encoder,
        hidden_states,
        attention_mask,
        kl_annealing_scheduler,
        kl_step,
        weighted_kl,
        lambda_kld,
        lambda_klg,
        kld,
        klg,
        training,
        kl_list_name=None,  
    ):
        src_key_padding_mask = ~(attention_mask.bool())
        hidden_states, attention, klg_vals, kld_vals, latent_dict = encoder(
            hidden_states, src_key_padding_mask=src_key_padding_mask, kl_list_name=kl_list_name
        )
        kl_factor = kl_annealing_scheduler(kl_step)
        kld_loss = weighted_mean(kl_list=kld_vals, weighted_mean=weighted_kl) * lambda_kld * kl_factor
        klg_loss = weighted_mean(kl_list=klg_vals, weighted_mean=weighted_kl) * lambda_klg * kl_factor

        # Add to total losses
        kld = kld + kld_loss
        klg = klg + klg_loss

        # Log to wandb
        if training and wandb.run is not None:
            try:
                wandb.log({
                    f"kl/kld_loss_nvib{kl_list_name}": kld_loss.item(),
                    f"kl/klg_loss_nvib{kl_list_name}": klg_loss.item(),
                    f"kl/total_kl_loss_nvib{kl_list_name}": (kld_loss + klg_loss).item(),
                    "kl/kl_annealing_factor": kl_factor,
                    "kl/step": kl_step
                })
            except Exception as e:
                print(f"Warning: Could not log KL metrics: {e}")
        return hidden_states, attention, kld, klg, latent_dict
    
def inference(
    base_model_path: str = f"/project/phan/tqn/Adapter/LLM-SplitModel/temp-with-Qwen-Llama-100k-retry/checkpoint-11000",  # the only required argument
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    prompter = Prompter(prompt_template_name)
    # Register the model classes
    AutoConfig.register("split", SplitConfig)
    AutoModel.register(SplitConfig, SplitModel)
    AutoModelForCausalLM.register(SplitConfig, SplitModelForCausalLM)
    
    print(f"Loading model from: {base_model_path}")
    
    # Load the model using from_pretrained
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        use_safetensors=True  # Use safetensors since that's what the checkpoint uses
    )

      
    # Ensure model is in evaluation mode
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Now test with the actual prompt
    prompt = """
    Write a python function to remove duplicate numbers from a given number of lists
    """
    prompt =  prompter.generate_prompt(
                   prompt,
                    "",
                    "",
                )
    temp = make_chat_prompt(prompt,instruction_prefix, response_prefix, tokenizer)
    print(temp)
    
    
    model_inputs = tokenizer([temp], return_tensors="pt").to(model.device)
    attention_mask = model_inputs.attention_mask
    
    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=attention_mask, max_new_tokens=1024)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(output_text)
    exit()


def main():
    inference()

if __name__ == "__main__":
    main()