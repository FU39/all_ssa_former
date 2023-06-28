import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from timm.models.layers import DropPath
from torch import distributed as dist
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

from deepspeed.ops.sparse_attention.matmul import MatMul
# from deepspeed.ops.sparse_attention import SparsityConfig


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class WarpFloatTensorBooling(Function):
    @staticmethod
    def forward(self, input):
        # self.save_for_backward(input.bool())
        output = input.bool().float()
        return output

    @staticmethod
    def backward(self, grad_output):
        # input_bool = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input_bool == False] = 0
        return grad_input


def warp_bool_float(input):
    # return WarpFloatTensorBooling.apply(input)
    return input.bool().float()


class MS_MLP_Linear(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        # self.fc1_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        # self.fc2_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, N, C = x.shape
        identity = x

        x = self.fc1_lif(x).flatten(0, 1)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_linear(x)
        x = (
            self.fc1_bn(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, self.c_hidden)
            .contiguous()
        )

        x = self.fc2_lif(x).flatten(0, 1)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_linear(x)
        x = (
            self.fc2_bn(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
            .contiguous()
        )

        x = x + identity

        return x, hook


class MS_SSA_Linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        dvs=False,
        layer=0
    ):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, detach_reset=True, backend="cupy"
        # )
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        # self.k_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, detach_reset=True, backend="cupy"
        # )
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        # self.v_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, detach_reset=True, backend="cupy"
        # )
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # self.attn_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        # )

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, detach_reset=True, backend="cupy"
        # )
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # self.shortcut_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, detach_reset=True, backend="cupy"
        # )
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.mode = mode
        self.layer = layer

        # # sparsity information
        # self.sparsity_config = SparsityConfig(num_heads=num_heads)

        # # initialize sparse layout and register as buffer
        # master_layout = self.sparsity_config.make_layout(max_seq_length)
        # self.register_buffer("master_layout", master_layout)
        # self._need_layout_synchronization = True

    # def get_layout(self, L):
    #     # if layout is never synchronized across GPUs, broadcast the layout from global rank 0
    #     if self._need_layout_synchronization and dist.is_initialized():
    #         dist.broadcast(self.master_layout, src=0)
    #         self._need_layout_synchronization = False

    #     if (L % self.sparsity_config.block != 0):
    #         raise ValueError(
    #             f'Sequence Length, {L}, needs to be dividable by Block size {self.sparsity_config.block}!'
    #         )

    #     num_blocks = L // self.sparsity_config.block
    #     return self.master_layout[..., :num_blocks, :num_blocks].cpu()  # layout needs to be a CPU tensor

    def forward(self, x, hook=None):
        T, B, N, C = x.shape
        identity = x
        attn = 0
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = (
            self.q_bn(q_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
            .contiguous()
        )
        q_linear_out = self.q_lif(q_linear_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_linear_out.detach()
        q = (
            q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = (
            self.k_bn(k_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, C, N)
            .contiguous()
        )
        k_linear_out = self.k_lif(k_linear_out)
        if self.dvs:
            k_linear_out = self.pool(k_linear_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_linear_out.detach()
        k = (
            k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = (
            self.v_bn(v_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, C, N)
            .contiguous()
        )
        v_linear_out = self.v_lif(v_linear_out)
        if self.dvs:
            v_linear_out = self.pool(v_linear_out)
        if hook is not None:
            hook[
                self._get_name() + str(self.layer) + "_v_lif"
                ] = v_linear_out.detach()
        v = (
            v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        if self.mode == "direct_xor":
            attn = 1 - torch.logical_xor(q.int(), k.int()).float()
            x = attn.mul(v)
        elif self.mode == "direct_matmul":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            x = attn @ v
        elif self.mode == "sparse_matmul":
            # TODO
            attn = (q @ k.transpose(-2, -1)) * self.scale
            x = attn @ v
        elif self.mode == "sparse_xor":
            attn = 1 - torch.logical_xor(q.int(), k.int()).float()
            sparsity_layout = self.get_layout(N)
            sparse_dot_dsd_nn = MatMul(
                sparsity_layout,
                self.sparsity_config.block,
                "dsd",
                trans_a=False,
                trans_b=False,
            )
            x = sparse_dot_dsd_nn(attn, v)

        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        if self.mode == "direct_matmul" or self.mode == "sparse_matmul":
            x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = (
            self.proj_bn(self.proj_linear(x).transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
        )

        x = x + identity
        return x, attn, hook


class MS_Block_Linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = MS_SSA_Linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            dvs=dvs,
            layer=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Linear(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, layer=layer
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn


class Channel_SSA_Linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        dvs=False,
        layer=0
    ):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm2d(dim)
        # spike_mode is lif
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # spike_mode is plif
        # self.q_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.k_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.v_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        # self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        # self.attn_lif = MultiStepParametricLIFNode(init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        # self.talking_heads_lif = MultiStepParametricLIFNode(
        #     init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        # )

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm2d(dim)

        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        # self.shortcut_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, N, C = x.shape
        identity = x
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = (
            self.q_bn(q_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
            .contiguous()
        )
        q_linear_out = self.q_lif(q_linear_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_linear_out.detach()
        q = (
            q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_linear_out = self.k_conv(x_for_qkv)
        k_linear_out = (
            self.k_bn(k_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, C, N)
            .contiguous()
        )
        k_linear_out = self.k_lif(k_linear_out)
        if self.dvs:
            k_linear_out = self.pool(k_linear_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_linear_out.detach()
        k = (
            k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = (
            self.v_bn(v_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, C, N)
            .contiguous()
        )
        v_linear_out = self.v_lif(v_linear_out)
        if self.dvs:
            v_linear_out = self.pool(v_linear_out)
        if hook is not None:
            hook[
                self._get_name() + str(self.layer) + "_v_lif"
                ] = v_linear_out.detach()
        v = (
            v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if self.dvs:
            kv = self.pool(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = x.flatten(0, 1)
        x = (
            self.proj_bn(self.proj_linear(x).transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
        )

        x = x + identity
        # x = identity - x
        return x, v, hook


class Spatial_SSA_Linear(Channel_SSA_Linear):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0,
        sr_ratio=1,
        mode="direct_xor",
        dvs=False,
        layer=0
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            sr_ratio,
            mode,
            dvs,
            layer
        )

    def forward(self, x, hook=None):
        T, B, N, C = x.shape
        identity = x
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = (
            self.q_bn(q_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
            .contiguous()
        )
        q_linear_out = self.q_lif(q_linear_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_linear_out.detach()
        q = (
            q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_linear_out = self.k_conv(x_for_qkv)
        k_linear_out = (
            self.k_bn(k_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, C, N)
            .contiguous()
        )
        k_linear_out = self.k_lif(k_linear_out)
        if self.dvs:
            k_linear_out = self.pool(k_linear_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_linear_out.detach()
        k = (
            k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = (
            self.v_bn(v_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, C, N)
            .contiguous()
        )
        v_linear_out = self.v_lif(v_linear_out)
        if self.dvs:
            v_linear_out = self.pool(v_linear_out)
        if hook is not None:
            hook[
                self._get_name() + str(self.layer) + "_v_lif"
                ] = v_linear_out.detach()
        v = (
            v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        kv = kv.sum(dim=-1, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if self.dvs:
            kv = self.pool(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = x.flatten(0, 1)
        x = (
            self.proj_bn(self.proj_linear(x).transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
        )

        x = x + identity
        # x = identity - x
        return x, v, hook


class Hybrid_Block_Linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.channel_attn = Channel_SSA_Linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            dvs=dvs,
            layer=layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.spatial_attn = Spatial_SSA_Linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            dvs=dvs,
            layer=layer
        )

    def forward(self, x, hook=None):
        x, attn, hook = self.channel_attn(x, hook=hook)
        x, attn, hook = self.spatial_attn(x, hook=hook)
        return x, attn, hook
