import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import wandb
from typing import Optional, List, Tuple, Union
from transformers import (
    GenerationMixin,
    PreTrainedModel,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from configuration_split import SplitConfig
from nvib.denoising_attention import DenoisingMultiheadAttention
from nvib.nvib_layer import Nvib
from nvib_selfattention.nvib_sa_transformer_encoder import (
    NVIBTransformerEncoder,
    NVIBTransformerEncoderLayer,
)
from update_causal_mask import _prepare_4d_causal_attention_mask_with_cache_position
from support_split_model import init_weights, weighted_mean
from kl_annealing import kl_annealing

class SplitModel(PreTrainedModel, GenerationMixin):
    config_class = SplitConfig
    def __init__(self, config):
        super().__init__(config)
        
class SplitModelForCausalLM(SplitModel):
    def __init__(self, config: SplitConfig):
        super().__init__(config)
        
        # Initialize base model with correct dtype from the start
        self.vocab_size = config.vocab_size
        # reference_model = AutoModelForCausalLM.from_pretrained(
        #     config.base_model_path, torch_dtype=torch.bfloat16, attn_implementation=config.attn_implementation, device_map="auto"
        # )
        # self.middle_model = (
        #     AutoModel.from_pretrained(config.middle_model_path, torch_dtype=torch.bfloat16, attn_implementation=config.attn_implementation, device_map="auto")
        #     if config.middle_model_path else None
        # )
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
        unused_layers = self.model.layers[config.enc_num_layers:-config.dec_num_layers]
        self.model.layers = nn.ModuleList(list(self.model.layers[:config.enc_num_layers]) + list(self.model.layers[-config.dec_num_layers:]))
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
        self.nvib_transformer_adapter1 = NVIBTransformerEncoder(
            encoder_layer=nvib_transformer_layer1, 
            num_layers=config.num_nvib_encoder_layers, 
            norm=encoder_norm1,
        )
        encoder_norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5, dtype=torch.bfloat16)  # Ensure LayerNorm is bfloat16
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
        self.kld = None
        self.klg = None
        del reference_model
        del unused_layers
        gc.collect()
        torch.cuda.empty_cache()
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
            # if attention_mask is not None:
            #     pad_mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)  # [batch, seq_len, 1]
            #     inputs_embeds = inputs_embeds * pad_mask
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        kld = torch.tensor(0.0, device=hidden_states.device)
        klg = torch.tensor(0.0, device=hidden_states.device)
        encoder_hidden_states = None

        # 1️⃣ First client encoder layers
        for encoder_layer in self.model.layers[: self.config.enc_num_layers]:
            layer_output = encoder_layer(
                hidden_states,
                attention_mask=None,  # for flash attention
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_output[0]
        
        # 2️⃣ NVIB1 adapter
        encoder_hidden_states = hidden_states
        hidden_states, _, kld, klg, _ = self.apply_nvib_adapter(
            self.nvib_transformer_adapter1, hidden_states, attention_mask,
            self.kl_annealing_scheduler, self.kl_step,
            self.weighted_kl, self.lambda_kld, self.lambda_klg,
            kld, klg, self.training, "klg1"
        )
        
        # 3️⃣ Middle model layers
        for middle_layer in self.middle_model.layers:
            hidden_states = middle_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )[0]

        hidden_states = self.middle_model.norm(hidden_states)
        hidden_states, _, kld, klg, _ = self.apply_nvib_adapter(
            self.nvib_transformer_adapter2, hidden_states, attention_mask,
            self.kl_annealing_scheduler, self.kl_step,
            self.weighted_kl, self.lambda_kld, self.lambda_klg,
            kld, klg, self.training, "klg2"
        )
        hidden_states = hidden_states + encoder_hidden_states
        
        # 4️⃣ Remaining client layers (after middle block)
        for decoder_layer in self.model.layers[-self.config.dec_num_layers:]:
            # NVIB2 adapter before last decoder layer
        
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )[0]
        
        # 5️⃣ Final normalization
        hidden_states = self.model.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # labels = labels.clone()
            # labels[attention_mask == 0] = -100
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
        kl_factor = kl_annealing_scheduler(kl_step) if training else 1.0
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
