import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, N, C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = (
            self.fc1_bn(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, self.c_hidden)
            .contiguous()
        )
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0, 1))
        x = (
            self.fc2_bn(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
            .contiguous()
        )
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        spike_mode="lif",
        attn_mode="direct_matmul",
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        )

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x):
        T, B, N, C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = (
            self.q_bn(q_linear_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
            .contiguous()
        )
        q_linear_out = self.q_lif(q_linear_out)
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
        v = (
            v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(
            self.proj_bn(self.proj_linear(x).transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, N, C)
        )
        return x, v


class SEW_Block_Linear(nn.Module):
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
        spike_mode="lif",
        attn_mode="direct_matmul",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            spike_mode=spike_mode,
            attn_mode=attn_mode,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x_attn, attn = self.attn(x)
        x = x + x_attn
        x = x + self.mlp(x)

        return x, attn
