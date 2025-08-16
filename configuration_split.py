# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Split model configuration derived from Qwen2 model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SplitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "split"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        # base_model_path: str = "/project/phan/codellama/FintunnedModel7B/CodeQwen_eps27_400k_tokenizerDP/checkpoint-82002",
        base_model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct",

        ##################### Qwen2.5-7B #####################
        attn_implementation="flash_attention_2",
        attention_dropout=0.0,
        bos_token_id= 151643,
        eos_token_id= 151645,
        hidden_act= "silu",
        hidden_size= 3584,
        initializer_range= 0.02,
        intermediate_size= 18944,
        max_position_embeddings= 32768,
        num_attention_heads= 28,
        num_hidden_layers= 28,
        num_key_value_heads= 4,
        rms_norm_eps= 1e-06,
        rope_theta= 1000000.0,
        rope_scaling=None,
        sliding_window= 131072, 
        use_sliding_window=False, 
        tie_word_embeddings= False,
        torch_dtype= "bfloat16",
        vocab_size= 152064,
        use_cache=False,
        layer_types=None, 
        max_window_layers=28,

        ##################### LLama3.1 #####################
        # attention_dropout=0.0,
        # bos_token_id= 128000,
        # eos_token_id= [
        #   128001,
        #   128008,
        #   128009
        # ],
        # hidden_act= "silu",
        # hidden_size= 4096,
        # initializer_range= 0.02,
        # intermediate_size= 14336,
        # max_position_embeddings= 131072,
        # mlp_bias= False,
        # num_attention_heads= 32,
        # num_hidden_layers= 32,
        # num_key_value_heads= 8,
        # pretraining_tp= 1,
        # rms_norm_eps= 1e-05,
        # rope_theta= 500000.0,
        # tie_word_embeddings= False,
        # torch_dtype= "bfloat16",
        # vocab_size= 128256,
        # use_cache=False,
        # rope_scaling= {
        # "factor": 8.0,
        # "low_freq_factor": 1.0,
        # "high_freq_factor": 4.0,
        # "original_max_position_embeddings": 8192,
        # "rope_type": "llama3"
        # },
        # use_sliding_window = False, 
        # sliding_windown = None,

        ##################### Qwen2.5-Coder-1.5B #####################
        # vocab_size=151936,
        # bos_token_id=151643,
        # eos_token_id=151645,
        # hidden_size=1536,
        # intermediate_size=8960,
        # num_hidden_layers=28,
        # num_attention_heads=12,
        # num_key_value_heads=2,
        # hidden_act="silu",
        # max_position_embeddings=32768,
        # initializer_range=0.02,
        # rms_norm_eps=1e-06,
        # use_cache=False,
        # tie_word_embeddings=False,
        # # rope_theta=10000.0,
        # rope_theta=1000000.0,
        # rope_scaling=None,
        # use_sliding_window=False,
        # sliding_window=32768,
        # # sliding_window=null,
        # max_window_layers=28,
        # attention_dropout=0.0,
        # layer_types=None,
        # torch_dtype="bfloat16",

        # split model params
        enc_num_layers: int = 1,
        dec_num_layers: int = 4,
        nhead: int = 1,

        # nvib params
        is_nvib: bool = True,
        dropout: float = 0.1,
        num_nvib_encoder_layers: int = 1,
        kappa: float = 1,
        delta: float = 0.4,
        weighted_kl: bool = True,
        lambda_kld: float = 0,
        lambda_klg: float = 0,
        # server's model params
        compress_dim: int = 4096,
        compress_intermediate_size: int = 14336,
        is_merge: bool = True,
        middle_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        # middle_model_path: str = "/project/phan/codellama/FintunnedModel7B/CodeQwen_eps27_400k_tokenizerDP/checkpoint-82002",
        **kwargs ,
    ):
        self.attn_implementation = attn_implementation
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.torch_dtype = torch_dtype
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        self.base_model_path = base_model_path
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.nhead = nhead
        self.is_nvib = is_nvib
        self.dropout = dropout
        self.num_nvib_encoder_layers = num_nvib_encoder_layers
        self.kappa = kappa
        self.delta = delta
        self.weighted_kl = weighted_kl
        self.lambda_kld = lambda_kld
        self.lambda_klg = lambda_klg
        self.compress_dim = compress_dim
        self.compress_intermediate_size = compress_intermediate_size
        self.is_merge = is_merge
        self.middle_model_path = middle_model_path
        
        # Add missing attributes that are referenced in the model
        self.output_attentions = False
        self.output_hidden_states = False

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
