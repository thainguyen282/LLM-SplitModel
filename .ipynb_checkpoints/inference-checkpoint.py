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

logger = logging.get_logger(__name__)

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
                    # Initialize alpha projection weights with smaller values
                    # Use Kaiming initialization with smaller gain
                    torch.nn.init.kaiming_normal_(param, a=0.1)
                    # Scale down the weights to prevent extreme values
                    with torch.no_grad():
                        param *= 0.1
                else:
                    torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

def weighted_mean(kl_list, weighted_mean=False):
    if weighted_mean:
        weights = [i for i in range(1, len(kl_list) + 1)]
        # weights = [(2**i) for i in range(0, len(kl_list))]
    else:  # Equal weighted Mean
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
            torch_dtype=torch.bfloat16
        )
        
        # Initialize middle model from config if specified
        if config.middle_model_path is not None:
            middle_config = AutoConfig.from_pretrained(config.middle_model_path)
            self.middle_model = AutoModel.from_config(
                middle_config,
                torch_dtype=torch.bfloat16
            )
        else:
            self.middle_model = None
        
        self.model = reference_model.model
        self.lm_head = reference_model.lm_head
        if self.lm_head.weight.data_ptr() == self.model.embed_tokens.weight.data_ptr():
            print("Breaking shared tensor between lm_head and embed_tokens")
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.clone())

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
                wait_before_warmup=300,
                end_of_warmup=600,
                # wait_before_warmup=0,
                # end_of_warmup=0,
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

############################################################# custom code start here #############################################################
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
            # print(f"layer {layer_type} {layer_idx}")
            if torch.isnan(hidden_states).any():
                print(f"WARNING: NaN before layer!")
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
                hidden_states, all_hidden_states, all_self_attns, next_decoder_cache = self.process_layer(
                    layer, hidden_states, output_hidden_states, output_attentions, use_cache, causal_mask, position_ids, past_key_values, all_hidden_states, all_self_attns, next_decoder_cache, position_embeddings, cache_position, is_middle_model=(layer_type == 'middle'),
                )
            if torch.isnan(hidden_states).any():
                print(f"WARNING: NaN after layer!")
        hidden_states = self.model.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            output = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return (loss,) + output if (loss := self.compute_loss(hidden_states, labels, kld, klg, logits_to_keep, **loss_kwargs)) else output

        logits = self.lm_head(hidden_states[:, slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)
            if self.lambda_klg != 0 or self.lambda_kld != 0: 
                loss = loss + kld + klg
            
        if not return_dict:
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
        memory, attention, klg_vals, kld_vals, latent_dict = encoder(
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
        return memory, attention, kld, klg, latent_dict

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
    
def inference(
    # model/data params
    # base_model_path: str = f"/project/phan/codellama/CodeQwen1.5-7B-Chat",  # the only required argument
    # base_model_path: str = f"/mmfs1/project/phan/tqn/Adapter/LLM-SplitModel/temp-with-nvib2-save/checkpoint-85440",  # the only required argument
    base_model_path: str = f"/project/phan/Adapter/LLM-SplitModel/temp-with-Qwen-Llama-final-2nd/checkpoint-4500",  # the only required argument
    # base_model_path = "/mmfs1/project/phan/codellama/FintunnedModel7B/CodeQwen_eps27_400k_tokenizerDP/checkpoint-82002",
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
    prompt = "Write a python code to sum two numbers"
    prompt =  prompter.generate_prompt(
                   prompt,
                    "",
                    "",
                )
    temp = make_chat_prompt(prompt,instruction_prefix, response_prefix, tokenizer)
    print(temp)
    
    
    model_inputs = tokenizer([temp], return_tensors="pt").to(model.device)
    attention_mask = model_inputs.attention_mask
    # model_inputs['labels'] = model_inputs['input_ids']
    # outputs = model(**model_inputs)
    # loss = outputs.loss
    # loss.backward()

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         grad_norm = param.grad.norm().item()
    #         print(f"{name}: grad norm = {grad_norm}")
            
    #         if torch.all(param.grad == 0):
    #             print(f"\033[91m{name}: gradient is exactly zero!\033[0m")
    # exit()
    
    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=attention_mask, max_new_tokens=1024)
        # The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output_text)
    exit()


def main():
    inference()

if __name__ == "__main__":
    main()