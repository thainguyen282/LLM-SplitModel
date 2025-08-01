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

wandb.init(project="split-model-with-nvib", name="split-model-with-nvib")

instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def make_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer: AutoTokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt

def get_code_completion(prefix, suffix):
    text = prompt = f"""<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"""
    #model.eval()
    return text

class GradientExtractionCallback(TrainerCallback):
    def __init__(self, log_path="gradients_log.jsonl", check_nan=True):
        self.log_path = log_path
        self.check_nan = check_nan

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Clear file at start
        with open(self.log_path, "w") as f:
            f.write("")  # Truncate file

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        grad_info = {
            "step": state.global_step,
            "gradients": {},
            "nan_detected": False,
            "inf_detected": False,
        }
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                norm = grad.norm().item()
                grad_info["gradients"][name] = norm 

                if self.check_nan:
                    if torch.isnan(grad).any():
                        grad_info["nan_detected"] = True
                    if torch.isinf(grad).any():
                        grad_info["inf_detected"] = True

        # Write to JSONL
        with open(self.log_path, "a") as f:
            f.write(json.dumps(grad_info) + "\n")

        # Optional: print if there's a problem
        if grad_info["nan_detected"] or grad_info["inf_detected"]:
            print(f"[Step {state.global_step}] ⚠️ Detected NaN or Inf in gradients!")

class KLStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        model.kl_step = state.global_step

class KLMetricsCallback(TrainerCallback):
    """Callback to log KL metrics and training statistics."""
    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        
        if wandb.run is not None and hasattr(model, 'kld') and hasattr(model, 'klg'):
            try:
                # Log total KL losses if they exist
                if model.kld is not None and model.klg is not None:
                    wandb.log({
                        "kl/total_kld": model.kld.item(),
                        "kl/total_klg": model.klg.item(),
                        "kl/total_combined": (model.kld + model.klg).item(),
                        "kl/step": state.global_step
                    })
                
                # Log KL annealing information
                if hasattr(model, 'kl_annealing_scheduler') and model.kl_annealing_scheduler is not None:
                    kl_factor = model.kl_annealing_scheduler(state.global_step)
                    wandb.log({
                        "kl/annealing_factor": kl_factor,
                        "kl/annealing_step": state.global_step,
                    })
            except Exception as e:
                print(f"Warning: Could not log KL metrics in callback: {e}")
    
def init_weights(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            if "alpha_proj" in name:
                # Initialize alpha projection bias to small positive values
                # This ensures log_alpha starts with reasonable values
                torch.nn.init.constant_(param, 0.1)  # Small positive bias
            else:
                torch.nn.init.zeros_(param)
        elif "weight" in name:
            if param.dim() > 1:
                if "nvib_layer" in name and "alpha_proj" in name:
                    torch.nn.init.xavier_uniform_(param)
                    with torch.no_grad():
                        param *= 0.1
                else:
                    torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

def weighted_mean(kl_list, weighted_mean=False):
    if weighted_mean:
        weights = [i for i in range(1, len(kl_list) + 1)]
    else: 
        weights = [1 for i in range(0, len(kl_list))]
    weights = [weight / (sum(weights)) for weight in weights]
    return sum([torch.mean(kl_layer) * weights[i] for i, kl_layer in enumerate(kl_list)])

class kl_annealing:
    def __init__(
        self,
        end_of_warmup,
        wait_before_warmup=0,
        annealing_value_start=0,
        annealing_value_end=1,
        type="linear",
    ):
        self.annealing_value_start = annealing_value_start
        self.annealing_value_end = annealing_value_end
        self.end_of_warmup = end_of_warmup
        self.type = type
        self.wait_before_warmup = wait_before_warmup

    def __call__(self, step):
        # Linear annealing
        if self.type == "linear":
            if step < self.wait_before_warmup:
                return self.annealing_value_start
            elif step < self.end_of_warmup:
                return (step - self.wait_before_warmup) / (
                    self.end_of_warmup - self.wait_before_warmup
                )
            else:
                return self.annealing_value_end
        else:
            # Constant
            return self.annealing_value_end

class SplitModel(PreTrainedModel, GenerationMixin):
    config_class = SplitConfig
    def __init__(self, config):
        super().__init__(config)
    

class MySplitModel(SplitModel):

    def __init__(self, config: SplitConfig):
        super().__init__(config)
        
        # Initialize base model with correct dtype from the start
        reference_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=torch.bfloat16  # Load with bfloat16 from the beginning
        )
        if config.middle_model_path is not None:
            middle_model_causal = AutoModelForCausalLM.from_pretrained(
                config.middle_model_path,
                torch_dtype=torch.bfloat16  # Load with bfloat16 from the beginning
            )
            self.lm_head = middle_model_causal.lm_head
            self.middle_model = middle_model_causal.model
        else:
            self.middle_model = None
            self.lm_head = reference_model.model

        self.model = reference_model.model
        
        if self.lm_head.weight.data_ptr() == self.model.embed_tokens.weight.data_ptr():
            print("Breaking shared tensor between lm_head and embed_tokens")
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.clone())

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.nhead = config.nhead

        # split model params
        self.enc_num_layers = config.enc_num_layers
        self.dec_num_layers = config.dec_num_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.nhead = config.nhead
        self.is_nvib = config.is_nvib
        self.dropout = config.dropout
        self.num_nvib_encoder_layers = config.num_nvib_encoder_layers
        self.kappa = config.kappa
        self.delta = config.delta
        self.weighted_kl = config.weighted_kl
        self.lambda_kld = config.lambda_kld
        self.lambda_klg = config.lambda_klg
        self.compress_dim = config.compress_dim
        self.compress_intermediate_size = config.compress_intermediate_size
        self.is_merge = config.is_merge
        

        # NVIB Transformer encoder layers setup with memory optimizations
        nvib_transformer_encoder_layer1 = NVIBTransformerEncoderLayer(
            in_dim=self.hidden_size,
            out_dim=self.compress_dim,
            nhead=self.nhead,
            dim_feedforward=self.compress_intermediate_size,
            dropout=self.dropout,
            activation="relu",
            kappa=self.kappa,
            delta=self.delta,
            batch_first=True,
            dtype=torch.bfloat16  # Ensure NVIB layers are created with bfloat16
        )
        # nvib_transformer_encoder_layer2 = NVIBTransformerEncoderLayer(
        #     in_dim=self.compress_dim,
        #     out_dim=self.hidden_size,
        #     nhead=self.nhead,
        #     dim_feedforward=self.intermediate_size,
        #     dropout=self.dropout,
        #     activation="relu",
        #     kappa=self.kappa,
        #     delta=self.delta,
        #     batch_first=True,
        #     dtype=torch.bfloat16  # Ensure NVIB layers are created with bfloat16
        # )
        encoder_norm1 = nn.LayerNorm(self.compress_dim, eps=1e-5, dtype=torch.bfloat16)  # Ensure LayerNorm is bfloat16
        # encoder_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-5, dtype=torch.bfloat16)  # Ensure LayerNorm is bfloat16
        self.nvib_transformer_encoder1 = NVIBTransformerEncoder(
            encoder_layer=nvib_transformer_encoder_layer1, 
            num_layers=self.num_nvib_encoder_layers, 
            norm=encoder_norm1,
        )
        # self.nvib_transformer_encoder2 = NVIBTransformerEncoder(
        #     encoder_layer=nvib_transformer_encoder_layer2, 
        #     num_layers=self.num_nvib_encoder_layers, 
        #     norm=encoder_norm2,
        # )
        
        # Add layer normalization for middle model
        # if self.is_merge:
        #     self.pre_middle_norm = nn.LayerNorm(self.compress_dim, eps=1e-5, dtype=torch.bfloat16)
        #     self.post_middle_norm = nn.LayerNorm(self.compress_dim, eps=1e-5, dtype=torch.bfloat16)
        # else:
        #     self.pre_middle_norm = None
        #     self.post_middle_norm = None
            
        self.kl_step = 0
        self.kl_annealing_scheduler = (
            kl_annealing(
                annealing_value_start=0,
                annealing_value_end=1,
                wait_before_warmup=1000 * 0.3,
                end_of_warmup=1000 * 0.6,
                # wait_before_warmup=0,
                # end_of_warmup=0,
                type="linear",
            )
            if self.is_nvib
            else None
        )

        init_weights(self.nvib_transformer_encoder1)
        # init_weights(self.nvib_transformer_encoder2)

        self.hidden_states = None
        
        # Initialize KL losses
        self.kld = None
        self.klg = None
        
        del reference_model
        if config.middle_model_path is not None:
            del middle_model_causal
        gc.collect()

    def process_layer(
        self,
        layer,
        hidden_states,
        output_hidden_states,
        output_attentions,
        use_cache,
        causal_mask,
        position_ids,
        past_key_values,
        output_hidden_states_tuple,
        output_attentions_tuple,
        next_decoder_cache,
        position_embeddings=None,
        cache_position=None,
        is_middle_model=False,
    ):
        """Helper function to process a single layer"""
        if output_hidden_states:
            output_hidden_states_tuple += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            assert not torch.isnan(layer_outputs[0]).any(), "nan after attention block"

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            output_attentions_tuple += (layer_outputs[1],)

        return hidden_states, output_hidden_states_tuple, output_attentions_tuple, next_decoder_cache

    def apply_nvib_encoder(
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
        """Helper function to apply NVIB encoder and handle KL losses"""
        src_key_padding_mask = ~(attention_mask.bool())
        memory, attention, klg_vals, kld_vals, latent_dict = encoder(
            hidden_states, src_key_padding_mask=src_key_padding_mask
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

        return memory, attention, kld, klg, latent_dict

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        if causal_mask is not None:
            # Fix extreme values that cause NaN
            if causal_mask.dtype == torch.bfloat16:
                # For bfloat16, clamp to safe range
                causal_mask = torch.clamp(causal_mask, min=-1e4, max=0)
            else:
                # For other dtypes, use standard range
                causal_mask = torch.clamp(causal_mask, min=-1e9, max=0)
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        kld = torch.tensor(0.0, device=hidden_states.device)
        klg = torch.tensor(0.0, device=hidden_states.device)

        # Create unified layer sequence with pointer-like approach
        layer_sequence = []
        
        # Add client model layers up to enc_num_layers
        for i in range(self.enc_num_layers):
            layer_sequence.append(('client', i, self.model.layers[i]))
        
        # Add first NVIB encoder
        if self.is_nvib:    
            layer_sequence.append(('nvib1', None, self.nvib_transformer_encoder1))
        
        # Add middle model layers if merging
        if self.is_merge:
            # layer_sequence.append(('pre_middle_norm', None, None))  # Placeholder for pre-middle norm
            for i, middle_layer in enumerate(self.middle_model.layers):
                layer_sequence.append(('middle', i, middle_layer))
            # layer_sequence.append(('post_middle_norm', None, None))  # Placeholder for post-middle norm
        
        # Add remaining client model layers
        # for i in range(self.enc_num_layers, self.num_hidden_layers):

            
        #     # Add second NVIB encoder at decoder start (if not merging)
        #     if self.is_merge:
        #         if i < self.num_hidden_layers - self.dec_num_layers:
        #             # if using mergedmodel, ignore middle layers
        #             continue
        #     if i == self.num_hidden_layers - self.dec_num_layers and self.is_nvib:
        #         layer_sequence.append(('nvib2', None, self.nvib_transformer_encoder2))

        #     layer_sequence.append(('client', i, self.model.layers[i]))
        
        encoder_hidden_states = None
        # Single loop through unified layer sequence
        for layer_type, layer_idx, layer in layer_sequence:
            # print(f"layer {layer_type} {layer_idx} with hidden_state size: {hidden_states.shape}")
            if torch.isnan(hidden_states).any():
                print(f"WARNING: NaN before layer!")
            if layer_type == 'nvib1':
                encoder_hidden_states = hidden_states
                memory, attention, kld, klg, latent_dict = self.apply_nvib_encoder(
                    layer, hidden_states, attention_mask, self.kl_annealing_scheduler, self.kl_step, self.weighted_kl, self.lambda_kld, self.lambda_klg, kld, klg, self.training, "klg1"
                )
                hidden_states = memory
            # elif layer_type == 'nvib2':
            #     memory, attention, kld, klg, latent_dict = self.apply_nvib_encoder(
            #         layer, hidden_states, attention_mask, self.kl_annealing_scheduler, self.kl_step, self.weighted_kl, self.lambda_kld, self.lambda_klg, kld, klg, self.training, "klg2"
            #     )
            #     hidden_states = memory + encoder_hidden_states
            #     # hidden_states = memory
            # elif layer_type == 'pre_middle_norm':
            #     hidden_states = self.pre_middle_norm(hidden_states)
            # elif layer_type == 'post_middle_norm':
            #     hidden_states = self.post_middle_norm(hidden_states)
            else:  # 'client' or 'middle' layers
                hidden_states, all_hidden_states, all_self_attns, next_decoder_cache = self.process_layer(
                    layer,
                    hidden_states,
                    output_hidden_states,
                    output_attentions,
                    use_cache,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                    next_decoder_cache,
                    position_embeddings,
                    cache_position,
                    is_middle_model=(layer_type == 'middle'),
                )
            if torch.isnan(hidden_states).any():
                print(f"WARNING: NaN after layer!")

        # hidden_states = self.model.norm(hidden_states)
        self.middle_model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)
            if self.lambda_klg != 0 and self.lambda_kld != 0: 
                loss = loss + kld + klg
        
        # if self.is_nvib:
        #     del memory, attention, latent_dict
            
        if not return_dict:
            # Fix: Return proper tuple of tensors, not tuple + dict
            output = (hidden_states, next_cache, all_hidden_states, all_self_attns)
            output = tuple(v for v in output if v is not None)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if causal_mask is not None:
            # Fix extreme values that cause NaN
            if causal_mask.dtype == torch.bfloat16:
                # For bfloat16, clamp to safe range
                causal_mask = torch.clamp(causal_mask, min=-1e4, max=0)     
            else:
                # For other dtypes, use standard range
                causal_mask = torch.clamp(causal_mask, min=-1e9, max=0)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: SplitConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device, dtype=dtype) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device, dtype=dtype) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask |= sliding_attend_mask
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


def train(
    # model/data params
    base_model_path: str = f"meta-llama/Llama-3.1-8B-Instruct",  # the only required argument
    # base_model_path: str = f"/project/phan/codellama/CodeQwen1.5-7B-Chat",  # the only required argument
    #    base_model_path: str = f"deepseek-ai/deepseek-coder-6.7b-instruct",  # the only required argument
    # base_model_path: str = f"/project/phan/codellama/FintunnedModel7B/CodeQwen_eps27_400k_tokenizerDP/checkpoint-82002", 
    # base_model_path: str = f"/mmfs1/project/phan/tqn/Adapter/LLM-SplitModel/temp-with-nvib2/checkpoint-55836",  # the only required argument
    # data_path: str = "OpenCoder-LLM/opc-sft-stage1",
    # split: str = "largescale_diverse_instruct",
    data_path: str = "iamtarun/python_code_instructions_18k_alpaca",
    # data_path: str = f"../datasets/PGCodeTraining68k.jsonl",
    output_dir: str = f"./temp-with-skip-connection-llama-Qwen-retry/",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1500,
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


    AutoConfig.register("split", SplitConfig)
    AutoModel.register(SplitConfig, SplitModel)
    AutoModelForCausalLM.register(SplitConfig, MySplitModel)    

    config = SplitConfig(
        base_model_path=base_model_path, 
    )
    tokenizer1 = AutoTokenizer.from_pretrained(config.base_model_path)
    tokenizer2 = AutoTokenizer.from_pretrained(config.middle_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(f"../dictionary")

    model = MySplitModel(config=config)
    model.to(device="cuda" if torch.cuda.is_available() else "cpu",dtype=torch.bfloat16)  # Only convert device, dtype is already correct
    model.train()
    model.config.pad_token_id = tokenizer2.pad_token_id
    model.config.eos_token_id = tokenizer2.eos_token_id
    model.config.bos_token_id = tokenizer2.bos_token_id
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer1.pad_token = tokenizer1.eos_token
    model.generation_config.pad_token_id = tokenizer2.bos_token_id
    model.generation_config.eos_token_id = config.eos_token_id
    model.generation_config.bos_token_id = tokenizer2.bos_token_id
    print(model.device)
    total_param = 0
    trainable_param = 0
    for param in model.parameters(): 
        param.requires_grad = False
        total_param += param.numel()
    for param in model.nvib_transformer_encoder1.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    # for param in model.nvib_transformer_encoder2.parameters(): 
    #     param.requires_grad = True
    #     trainable_param += param.numel()
    for param in model.lm_head.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.layers[:model.config.enc_num_layers].parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.middle_model.layers[-model.config.dec_num_layers:].parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    # for param in model.pre_middle_norm.parameters(): 
    #     param.requires_grad = True
    #     trainable_param += param.numel()
    # for param in model.post_middle_norm.parameters(): 
    #     param.requires_grad = True
    #     trainable_param += param.numel()

    model.model.embed_tokens.weight.requires_grad_(True)

    print(f'Total Parameters: {total_param:,}')
    print(f'Trainable Parameters: {trainable_param:,}')
    print(f'Non-trainable Parameters: {total_param - trainable_param:,}')
    
    # Print percentage of trainable parameters
    print(f'Percentage of trainable parameters: {100 * trainable_param / total_param:.2f}%')
        
    tokenizer1.padding_side = "left"  # Allow batched inference  
        
    # def tokenize(prompt, add_eos_token=True):
    #     # there's probably a way to do this with the tokenizer settings
    #     # but again, gotta move fast
    #     result = tokenizer1(
    #         prompt,
    #         truncation=True,
    #         max_length=cutoff_len,
    #         # padding=False,
    #         padding="max_length",
    #         return_tensors=None,
    #     )
    #     labels = tokenizer2(
    #         prompt,
    #         truncation=True,
    #         max_length=cutoff_len,
    #         # padding=False,
    #         padding="max_length",
    #         return_tensors=None,
    #     )["input_ids"]
    #     if (
    #         result["input_ids"][-1] != tokenizer1.eos_token_id
    #         and len(result["input_ids"]) < cutoff_len
    #         and add_eos_token
    #     ):
    #         result["input_ids"].append(tokenizer1.eos_token_id)
    #         result["attention_mask"].append(1)
    #         labels.append(tokenizer2.eos_token_id)
        
    #     # result["labels"] = result["input_ids"].copy()
    #     result["labels"] = labels

    #     return result

    def custom_tokenize(prompt, output, add_eos_token=True):
        # Tokenize the prompt using tokenizer1
        result1 = tokenizer1(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        # Calculate available length for output tokens
        remaining_len = cutoff_len - len(result1["input_ids"])
        
        # Tokenize the output using tokenizer2
        result2 = tokenizer2(
            output,
            truncation=True,
            max_length=remaining_len,
            padding=False,
            return_tensors=None,
        )
        
        # Concatenate input_ids and attention_mask from both tokenizations
        input_ids = result1["input_ids"] + result2["input_ids"]
        attention_mask = result1["attention_mask"] + result2["attention_mask"]
        # print(f"input_ids shape: {len(input_ids)}")
        # print(f"attention_mask shape: {len(attention_mask)}")
        # exit()
        
        # Optionally add EOS token if it's not already present
        if (
            add_eos_token
            and input_ids[-1] != tokenizer2.eos_token_id
            and len(input_ids) < cutoff_len
        ):
            input_ids.append(tokenizer2.eos_token_id)
            attention_mask.append(1)
            # Set labels to be a copy of input_id
            labels = input_ids.copy()
            return {
               "input_ids": torch.tensor(input_ids).unsqueeze(0),  # add batch dimension
               "attention_mask": torch.tensor(attention_mask).unsqueeze(0),
               "labels": torch.tensor(labels).unsqueeze(0),
            }

    template = "    \"\"\"{}\"\"\"\n"
    def generate_and_tokenize_prompt_auto_completion(data_point):
        if data_point["input"] != "":
            system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            
        temp_prompt =  data_point['output'].split(":")[0] + ":\n"+template.format(data_point['instruction'])
        
        # messages = make_chat_prompt(full_prompt[len(system)+2:],instruction_prefix, response_prefix, tokenizer)
        output = data_point["output"]
        temp_tokenizer = tokenizer1(
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
                prefix = tokenizer1.batch_decode([temp_tokenizer['input_ids'][random_numbers[0]]], skip_special_tokens=False)[0]
            else:
                #prefix = temp_prompt + tokenizer.batch_decode([temp_tokenizer['input_ids'][:random_numbers[0]]], skip_special_tokens=False)[0]
                prefix = tokenizer1.batch_decode([temp_tokenizer['input_ids'][:random_numbers[0]]], skip_special_tokens=False)[0]
            
            if random_numbers[1] == len(temp_tokenizer['input_ids']):
                suffix = "\"\"\""
            else:
                suffix = tokenizer1.batch_decode([temp_tokenizer['input_ids'][random_numbers[1]:]], skip_special_tokens=False)[0]
        
            temp_prompt = get_code_completion(prefix, suffix)
            #print(temp_prompt)
            temp_prompt = prompter.generate_prompt(
                            temp_prompt,
                            data_point["input"],
                            "",
                        )
            
            messages = make_chat_prompt(temp_prompt,instruction_prefix, response_prefix, tokenizer1)
            
            
            middle = tokenizer1.batch_decode([temp_tokenizer['input_ids'][random_numbers[0]:random_numbers[1]]], skip_special_tokens=False)[0]
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
        
        messages = make_chat_prompt(full_prompt[len(system)+2:],instruction_prefix, response_prefix, tokenizer1)
        
        output = data_point["output"]
        # messages += output
        # print(messages)
        # exit()
        
        # tokenized_full_prompt = tokenize(messages)
        tokenized_full_prompt = custom_tokenize(messages, output)
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
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    # data['train'] = Dataset.from_dict(data['train'][0])
    

    #Loading Disk Datasets
    
    # data = load_from_disk(f'../datasets/PGCodeTraining400k')
    
    
    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         #model = set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")
      # Be more transparent about the % of trainable params.
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        # train_data_generation = (
        #     train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # train_data_auto_completion = (
        #     train_val["train"].shuffle().map(generate_and_tokenize_prompt_auto_completion)
        # )
        # train_data = concatenate_datasets([train_data_generation, train_data_auto_completion]).shuffle()


        # val_data_generation = (
        #     train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # val_data_auto_completion = (
        #     train_val["test"].shuffle().map(generate_and_tokenize_prompt_auto_completion)
        # )
        # val_data = concatenate_datasets([val_data_generation, val_data_auto_completion]).shuffle()

        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
        
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    print(torch.cuda.device_count())
    #model.parallelize()
    #print(ddp)
    #if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True
    # Save train and validation data to JSON files
    train_data.to_json("train_data_llama2.json")
    val_data.to_json("val_data_llama2.json")
    # print("Saved training and validation data to JSON files")

    
    # model.train()
    # prompt = "Write a Python function that takes an array of integers and finds the maximum difference between any two elements in the array"
    # prompt =  prompter.generate_prompt(
    #                prompt,
    #                 "",
    #                 "",
    #             )
    # temp = make_chat_prompt(prompt,instruction_prefix, response_prefix, tokenizer1)
    
    # model_inputs = tokenizer1([temp], return_tensors="pt", padding="max_length", max_length=cutoff_len).to(model.device)
    # attention_mask = model_inputs.attention_mask
    # outputs = model(**model_inputs)
    # loss = outputs.loss
    # loss.backward()
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         grad_norm = param.grad.norm().item()
    #         print(f"{name}: grad norm = {grad_norm}")
            
    #         if torch.all(param.grad == 0):
    #             print(f"\033[91m{name}: gradient is exactly zero!\033[0m")
    # streamer = TextStreamer(tokenizer,skip_prompt=True)
    # Use `max_new_tokens` to control the maximum output length.
    # with torch.no_grad():
    #     generated_ids = model.generate(model_inputs.input_ids,attention_mask=attention_mask, max_new_tokens=1024)
    #     # The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
        
    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
    
    # output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(output_text)
    # exit()
    
    # train_data = load_dataset('json', data_files="train_data_llama.json")["train"]
    # val_data = load_dataset('json', data_files="val_data_llama.json")["train"]

    trainer = transformers.Trainer(
        model=model,
        tokenizer = tokenizer2,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=750,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,  # Disable fp16 since model uses bfloat16
            bf16=True,   # Use bf16 for mixed precision with bfloat16 models
            logging_steps=20,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps= 346 if val_set_size > 0 else None,
            save_steps=346,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            # Add gradient clipping for numerical stability
            max_grad_norm=1.0,
        ),
        callbacks=[KLStepCallback(), KLMetricsCallback(), GradientExtractionCallback(log_path="/project/phan/tqn/Adapter/LLM-SplitModel/gradients_log.jsonl")],
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
    )
    
    #model.config.use_cache = False

    #old_state_dict = model.state_dict
    #print(old_state_dict)
    
    #model.state_dict = (
    #    lambda self, *_, **__: get_peft_model_state_dict(
    #        self, old_state_dict()
    #    )
    #).__get__(model, type(model))

    # trainer.train()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    model.save_pretrained(output_dir, safe_serialization=False)
    
    
   
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)

