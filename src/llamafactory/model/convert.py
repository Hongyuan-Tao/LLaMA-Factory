from functools import partial
from tqdm import tqdm
import torch.nn as nn
from .fla import layers
from ..extras import logging
import importlib

logger = logging.get_logger(__name__)

def convert_attention(model: nn.Module, 
                      finetuning_args: dict):
    """
    Call to convert all attention layers
    """
    all_layers = traverse_layers(model)

    for layer_idx, layer in enumerate(tqdm(all_layers, desc='Converting attentions...')):
        if layer_idx not in finetuning_args.softmax_attention:
            try:
                layer.self_attn = convert_quadratic_to_linear(layer, finetuning_args.mixer, **finetuning_args.mixer_config)
                layer.self_attn.converted = True
            except:
                layer.attention = convert_quadratic_to_linear(layer, finetuning_args.mixer, **finetuning_args.mixer_config)
                layer.attention.converted = True

        else:  # Freeze any preserved softmax attention layers
            for p in layer.parameters():
                p.requires_grad = False
                    
    return model

def traverse_layers(model: nn.Module, verbose: bool = False):
    """
    Return list of model layers
    """
    try:
        all_layers = model.model.layers
        if verbose:
            print('-> Loading from model.model.layers')
    except AttributeError as e: # if base model
        if verbose:
            print(e)
        try:
            all_layers = model.layers
            if verbose:
                print('-> Loading from model.layers')
        except AttributeError as e1:  # If we make a PEFT model
            if verbose:
                print(e1)
            try:
                all_layers = model.base_model.model.model.layers
                if verbose:
                    print('-> Loading from model.base_model.model.model.layers')
            except AttributeError as e2:
                if verbose:
                    print(e2)
                all_layers = model.language_model.model.layers
                if verbose:
                    print('-> Loading from model.language_model.model.layers')

    return all_layers

def convert_quadratic_to_linear(layer: nn.Module, mixer: str, **kwargs):
    """
    Converts a single layer's attention layer as specified by attention_config
    """

    ParentClass = getattr(layers, mixer, None)

    class MixerWrapper(ParentClass):
        def __init__(self, layer, mixer, **kwargs):
            if mixer == "Mamba2":
                try:
                    super().__init__(
                                    hidden_size = layer.self_attn.hidden_size,
                                    num_heads = layer.self_attn.num_heads,
                                    head_dim = layer.self_attn.head_dim,
                                    num_key_value_heads = layer.self_attn.num_key_value_heads,
                                    num_key_value_groups = layer.self_attn.num_key_value_groups,
                                    layer_idx = layer.self_attn.layer_idx,
                                        **kwargs
                                        )
                    self.q_proj = layer.self_attn.q_proj
                    self.k_proj = layer.self_attn.k_proj
                    self.v_proj = layer.self_attn.v_proj
                    self.o_proj = layer.self_attn.o_proj
                    self.device = self.q_proj.weight.device
                    self.dtype = self.q_proj.weight.dtype
                    for name, param in self.named_parameters():
                        param.data = param.data.to(device=self.device, dtype=self.dtype)
                
                except AttributeError as e:
                    print(e)

            if mixer == "Mamba2_new":
                try:
                    super().__init__(
                                    hidden_size = layer.self_attn.hidden_size,
                                    num_heads = layer.self_attn.num_heads,
                                    head_dim = layer.self_attn.head_dim,
                                    layer_idx = layer.self_attn.layer_idx,
                                        **kwargs
                                        )
                    self.q_proj = layer.self_attn.q_proj
                    k_weight = layer.self_attn.k_proj.weight.data  # (num_kv_heads * head_dim, hidden_size)
                    v_weight = layer.self_attn.v_proj.weight.data  # (num_kv_heads * head_dim, hidden_size)
                    # 重塑权重以便重复：(num_kv_heads, head_dim, hidden_size)
                    k_weight = k_weight.view(layer.self_attn.num_key_value_heads, self.head_dim, self.hidden_size)
                    v_weight = v_weight.view(layer.self_attn.num_key_value_heads, self.head_dim, self.hidden_size)

                    # 沿 num_heads 维度重复 n_rep 次
                    k_weight = k_weight.repeat(layer.self_attn.num_key_value_groups, 1, 1)  # (num_heads, head_dim, hidden_size)
                    v_weight = v_weight.repeat(layer.self_attn.num_key_value_groups, 1, 1)  # (num_heads, head_dim, hidden_size)

                    # 重新展平：(num_heads * head_dim, hidden_size)
                    k_weight = k_weight.reshape(self.num_heads * self.head_dim, self.hidden_size)
                    v_weight = v_weight.reshape(self.num_heads * self.head_dim, self.hidden_size)

                    self.k_proj.weight.data = k_weight
                    self.v_proj.weight.data = v_weight

                    if layer.self_attn.k_proj.bias is not None:
                        k_bias = layer.self_attn.k_proj.bias.data  
                        v_bias = layer.self_attn.v_proj.bias.data  
                        k_bias = k_bias.view(layer.self_attn.num_key_value_heads, self.head_dim)  
                        v_bias = v_bias.view(layer.self_attn.num_key_value_heads, self.head_dim)  
                        k_bias = k_bias.repeat(layer.self_attn.num_key_value_groups, 1) 
                        v_bias = v_bias.repeat(layer.self_attn.num_key_value_groups, 1)  
                        k_bias = k_bias.reshape(self.num_heads * self.head_dim)  
                        v_bias = v_bias.reshape(self.num_heads * self.head_dim) 
                        self.k_proj.bias.data = k_bias
                        self.v_proj.bias.data = v_bias

                    self.o_proj = layer.self_attn.o_proj
                    self.device = self.q_proj.weight.device
                    self.dtype = self.q_proj.weight.dtype
                    
                    for name, param in self.named_parameters():
                        param.data = param.data.to(device=self.device, dtype=self.dtype)
                
                except AttributeError as e:
                    print(e)


            if mixer == "TTTWrapper":
                try:
                    super().__init__(
                                    hidden_size = layer.self_attn.hidden_size,
                                    num_heads = layer.self_attn.num_heads,
                                    **kwargs
                                        )
                    self.ttt.wq = layer.self_attn.q_proj
                    k_weight = layer.self_attn.k_proj.weight.data  # (num_kv_heads * head_dim, hidden_size)
                    v_weight = layer.self_attn.v_proj.weight.data  # (num_kv_heads * head_dim, hidden_size)
                    # 重塑权重以便重复：(num_kv_heads, head_dim, hidden_size)
                    k_weight = k_weight.view(layer.self_attn.num_key_value_heads, layer.self_attn.head_dim, layer.self_attn.hidden_size)
                    v_weight = v_weight.view(layer.self_attn.num_key_value_heads, layer.self_attn.head_dim, layer.self_attn.hidden_size)

                    # 沿 num_heads 维度重复 n_rep 次
                    k_weight = k_weight.repeat(layer.self_attn.num_key_value_groups, 1, 1)  # (num_heads, head_dim, hidden_size)
                    v_weight = v_weight.repeat(layer.self_attn.num_key_value_groups, 1, 1)  # (num_heads, head_dim, hidden_size)

                    # 重新展平：(num_heads * head_dim, hidden_size)
                    k_weight = k_weight.reshape(layer.self_attn.num_heads * layer.self_attn.head_dim, layer.self_attn.hidden_size)
                    v_weight = v_weight.reshape(layer.self_attn.num_heads * layer.self_attn.head_dim, layer.self_attn.hidden_size)

                    self.ttt.wk.weight.data = k_weight
                    self.ttt.wv.weight.data = v_weight

                    if layer.self_attn.k_proj.bias is not None:
                        k_bias = layer.self_attn.k_proj.bias.data  
                        v_bias = layer.self_attn.v_proj.bias.data  
                        k_bias = k_bias.view(layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)  
                        v_bias = v_bias.view(layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)  
                        k_bias = k_bias.repeat(layer.self_attn.num_key_value_groups, 1) 
                        v_bias = v_bias.repeat(layer.self_attn.num_key_value_groups, 1)  
                        k_bias = k_bias.reshape(layer.self_attn.num_heads * layer.self_attn.head_dim)  
                        v_bias = v_bias.reshape(layer.self_attn.num_heads * layer.self_attn.head_dim) 
                        self.ttt.wk.bias.data = k_bias
                        self.ttt.wv.bias.data = v_bias

                    self.ttt.wo = layer.self_attn.o_proj
                    self.device = self.ttt.wq.weight.device
                    self.dtype = self.ttt.wq.weight.dtype
                    
                    for name, param in self.named_parameters():
                        param.data = param.data.to(device=self.device, dtype=self.dtype)
                
                except AttributeError as e:
                    print(e)
                    
            if mixer == "Qwen2_5_VLSdpaAttention":
                try:
                    super().__init__(
                                    config = layer.self_attn.config,
                                    layer_idx = layer.self_attn.layer_idx,
                                        )
                    
                    self.o_proj = layer.self_attn.o_proj
                    self.device = layer.self_attn.q_proj.weight.device
                    self.dtype = layer.self_attn.q_proj.weight.dtype
                    
                    for name, param in self.named_parameters():
                        param.data = param.data.to(device=self.device, dtype=self.dtype)

                except AttributeError as e:
                    print(e)

            if mixer == "GatedDeltaNet":
                try:
                    super().__init__(
                                    hidden_size = layer.self_attn.hidden_size,
                                    num_heads = layer.self_attn.num_heads,
                                    head_dim = layer.self_attn.head_dim,
                                    expand_v = 1,
                                    layer_idx = layer.self_attn.layer_idx,
                                        **kwargs
                                        )
                    
                    self.o_proj = layer.self_attn.o_proj
                    self.device = layer.self_attn.q_proj.weight.device
                    self.dtype = layer.self_attn.q_proj.weight.dtype
                    
                    for name, param in self.named_parameters():
                        param.data = param.data.to(device=self.device, dtype=self.dtype)
                
                except AttributeError as e:
                    print(e)

            #等待进一步补充...

    return MixerWrapper(layer, mixer, **kwargs) 



    """return get_attention(**attention_config)(
        base_attn=layer.self_attn,
        layer_idx=layer.self_attn.layer_idx,  # Transformers v4.36
        max_layer_idx=len(layers) - 1,
        train_attention=train_attention,
        remove_base_attn=remove_base_attn,
        use_D=attention_config.mamba2.use_D,
        use_qknorm=attention_config.mamba2.use_qknorm,
        use_conv=attention_config.mamba2.use_conv,
        use_gnorm=attention_config.mamba2.use_gnorm,
        use_A=attention_config.mamba2.use_A,
        inherit_qkv=attention_config.mamba2.inherit_qkv,
        mimic_init=attention_config.mamba2.mimic_init,
        stage1=attention_config.stage1,
        stage2=attention_config.stage2,
    )"""