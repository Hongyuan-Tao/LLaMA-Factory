import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from typing import List, Tuple, Optional

from ttt.models.cogvideo.utils import (SequenceMetadata, full_tensor,
                                       place_into, shard_tensor, to_local)
from ttt.models.ssm.linear_triton import TritonLinear
from ttt.models.ssm.mlp_tk import TkMLP
from ttt.models.ssm.ops import ttt_linear, ttt_mlp

class TTTWrapper(nn.Module):
    def __init__(self, 
                 hidden_size: int = 1024,  ###
                 num_heads: int = 32,  ###
                 mini_batch_size: int = 16,  ###
                 ttt_base_lr: float = 1.0,
                 scan_checkpoint_group_size: int = 16, 
                 use_kernel: bool = True,  
                 ssm_layer: str = "ttt_linear", 
                 ):
        super().__init__()

        self.model_dim = hidden_size
        self.num_heads = num_heads
        self.ssm_layer = ssm_layer
        #self.latent_height = config.latent_height
        #self.latent_width = config.latent_width

        if self.ssm_layer == "ttt_linear":
            self.ttt = TTTLinear(hidden_size, num_heads, mini_batch_size, ttt_base_lr, scan_checkpoint_group_size, use_kernel)
        elif self.ssm_layer == "ttt_mlp":
            self.ttt = TTTMLP(hidden_size, num_heads, mini_batch_size, ttt_base_lr, scan_checkpoint_group_size, use_kernel)
        else:
            raise TypeError(f"No ttt layer of type {self.ssm_layer}")

    def init_freqs(self):
        self.freqs_cis.copy_(self._precompute_freqs_cis_3d())

    def forward(self, 
                hidden_states: torch.Tensor,
                vision_patch_indices: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # "legacy" cache approach
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,):
        return self.ttt(hidden_states), None, None
    
class TTTBase(nn.Module):
    def __init__(self,
                 model_dim: int = 1024,
                 num_heads: int = 32,
                 mini_batch_size: int = 16,
                 ttt_base_lr: float = 1.0,
                 scan_checkpoint_group_size: int = 16, 
                 ):
        super().__init__()
        self.width = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.mini_batch_size = mini_batch_size

        self.ttt_base_lr = ttt_base_lr
        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        self.tp_mesh: None | DeviceMesh = None

        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    # We must reinitialize after meta initialization
    def init_weights(self):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)

        self.post_norm.reset_parameters()
        nn.init.ones_(self.ttt_norm_weight.data)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.normal_(self.learnable_ttt_lr_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias)

    def _init_qkvo_proj(self):
        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )

        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        self.tp_mesh = tp_mesh

        self.ttt_norm_weight = nn.Parameter(distribute_tensor(self.ttt_norm_weight, tp_mesh, [Shard(0)]))
        self.ttt_norm_bias = nn.Parameter(distribute_tensor(self.ttt_norm_bias, tp_mesh, [Shard(0)]))

        self.learnable_ttt_lr_weight = nn.Parameter(
            distribute_tensor(self.learnable_ttt_lr_weight, tp_mesh, [Replicate()])
        )
        self.learnable_ttt_lr_bias = nn.Parameter(distribute_tensor(self.learnable_ttt_lr_bias, tp_mesh, [Replicate()]))

    def shard_inputs(self, inputs):
        assert self.tp_mesh is not None, "Tensor parallel mesh must be initialized before sharding inputs."

        for key in inputs:
            assert inputs[key].shape[1] == self.num_heads, "Sharding is only supported on the head dimension."
            inputs[key] = shard_tensor(inputs[key], self.tp_mesh, dim=1)

        return inputs

    @torch.compile
    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )
        return XQ, XK, XV

    @torch.compile
    def get_eta(self, X):
        learnable_ttt_lr_weight = full_tensor(self.learnable_ttt_lr_weight)
        learnable_ttt_lr_bias = full_tensor(self.learnable_ttt_lr_bias)

        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, learnable_ttt_lr_weight) + learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )  # [B,nc,cs,c] @ [nh,1,c] -> [B,nh,nc,cs,1] + [1,nh,1,1,1] -> [B,nh,nc,cs,1]

        ttt_lr = F.sigmoid(ttt_lr)  # [B,H,nc,K,1]

        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        return self.ttt_base_lr * ttt_lr / self.head_dim

    @torch.compile
    def ln_reconstruction_target(self, XV, XK):
        XV = XV - XK
        eps = 1e-8
        # Compute mean and std over the head dimension (last dimension)
        mean = XV.mean(dim=-1, keepdim=True)
        std = place_into(to_local(XV).std(dim=-1, keepdim=True), XV)

        # Normalize
        XV = (XV - mean) / (std + eps)

        # Apply per-head weight and bias.
        # self.ttt_norm_weight and self.ttt_norm_bias have shape [num_heads, head_dim].
        # We unsqueeze to make them broadcastable with XV_norm which is [B, L, num_heads, head_dim].
        XV = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0) * XV + self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)

        return XV + XK

    @torch.compile
    def reshape_to_mini_batch(self, X, XQ, XK, XV):
        B, L = X.shape[:2]
        num_mini_batch = L // self.mini_batch_size

        XQ, XK, XV = XQ.transpose(1, 2), XK.transpose(1, 2), XV.transpose(1, 2)

        X = X.reshape(B, num_mini_batch, self.mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)

        return X, XQ, XK, XV

    def process_input(self, hidden_states: torch.Tensor):
        B, L, D = hidden_states.shape
        mini_batch_size = self.mini_batch_size

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        pad_len = (mini_batch_size - (L % mini_batch_size)) % mini_batch_size  # 如果L已经是倍数，pad_len为0

        if pad_len > 0:
            pad_tensor = XQ.new_zeros(B, pad_len, D)  # [1, pad_len, d]
            hidden_states = torch.cat([hidden_states, pad_tensor], dim=1)
            XQ = torch.cat([XQ, pad_tensor], dim=1)
            XK = torch.cat([XK, pad_tensor], dim=1)
            XV = torch.cat([XV, pad_tensor], dim=1)
        
        B, L = XQ.shape[:2]

        XQ = XQ.view(B, L, -1, self.head_dim)
        XK = XK.view(B, L, -1, self.head_dim)
        XV = XV.view(B, L, -1, self.head_dim)

        # L2 Norm
        XQ = place_into(torch.nn.functional.normalize(to_local(XQ), p=2, dim=-1), XQ)
        XK = place_into(torch.nn.functional.normalize(to_local(XK), p=2, dim=-1), XK)

        XV = self.ln_reconstruction_target(XV, XK)

        hidden_states, XQ, XK, XV = self.reshape_to_mini_batch(hidden_states, XQ, XK, XV)

        ttt_lr_eta = self.get_eta(hidden_states)

        # We do not use token_eta for non-causal chunks
        eta = 1 / mini_batch_size * ttt_lr_eta.repeat(1, 1, 1, mini_batch_size, 1)

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
        }

        if self.tp_mesh is not None:
            inputs = self.shard_inputs(inputs)

        return inputs

    def ttt(
        self,
        inputs,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):

        original_len = hidden_states.shape[1]

        hidden_states = self.ttt(self.process_input(hidden_states))  # shape: [1, L+pad_len, d]

        hidden_states = hidden_states[:, :original_len, :]

        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.wo(hidden_states)

        hidden_states = full_tensor(hidden_states)

        return hidden_states


class TTTLinear(TTTBase):
    def __init__(self, 
                 model_dim: int = 1024,
                 num_heads: int = 32,
                 mini_batch_size: int = 16,
                 ttt_base_lr: float = 1.0,
                 scan_checkpoint_group_size: int = 16, 
                 use_kernel: bool = True):
        super().__init__(model_dim, num_heads, mini_batch_size, ttt_base_lr, scan_checkpoint_group_size)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # For acceleration
        self.use_kernel = use_kernel

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTLinear without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))

        TritonLinear.sharded_mode = True

    def ttt(self, inputs):
        B = inputs["XV"].shape[0]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        num_mini_batch = inputs["XV"].shape[2]

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.use_kernel:
            XQW_batch = TritonLinear.apply(
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                inputs["XQ"],
                inputs["XV"],
                inputs["XK"],
                inputs["eta"],
                checkpoint_group_size,
            )

            XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        else:
            XQW_batch = ttt_linear(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                checkpoint_group_size,
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch


class TTTMLP(TTTBase):
    def __init__(self, 
                 model_dim: int = 1024,
                 num_heads: int = 32,
                 mini_batch_size: int = 16,
                 ttt_base_lr: float = 1.0,
                 scan_checkpoint_group_size: int = 16, 
                 use_kernel: bool = True):
        super().__init__(model_dim, num_heads, mini_batch_size, ttt_base_lr, scan_checkpoint_group_size)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        self.use_kernel = use_kernel

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTMLP without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))
        self.W2 = nn.Parameter(distribute_tensor(self.W2, tp_mesh, [Shard(0)]))
        self.b2 = nn.Parameter(distribute_tensor(self.b2, tp_mesh, [Shard(0)]))

        TkMLP.sharded_mode = True

    def ttt(self, inputs):
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.use_kernel:
            XQW_batch = TkMLP.apply(
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states,
                b2_states,
                inputs["XQ"],
                inputs["XV"],
                inputs["XK"],
                inputs["eta"],
                checkpoint_group_size,
            )

            XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        else:
            XQW_batch = ttt_mlp(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states,
                b2_states,
                checkpoint_group_size,
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch