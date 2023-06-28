import torch
import torch.nn as nn
from torch.autograd import Function

# from spikingjelly.activation_based import surrogate, base, neuron
import math
from typing import Any, Optional, Tuple


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.gt(0.0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Derivative of 1/pi * arctan(pi * x) + 0.5
        spike_pseudo_grad = 1.0 / (math.pi * math.pi * x * x + 1.0)
        return grad_input * spike_pseudo_grad


def spike(x: torch.Tensor) -> torch.Tensor:
    return SpikeFunction.apply(x)


class ALIF(nn.Module):  # ALIF no reset
    def __init__(self, r_min, r_max, channels, thresh=1.0):
        super().__init__()

        # Shared between neurons in a layer
        self.init_flag = True
        self.r_min = r_min
        self.r_max = r_max
        # Just for convenience
        self.thresh = thresh
        A_log = self.init_recurrent(shape=[channels, 1, 1]).requires_grad_()
        self.register_parameter("A_log", nn.Parameter(A_log))

    def init_recurrent(self, shape):
        u = torch.rand(shape)
        nu_log = torch.log(
            -0.5 * torch.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2)
        )
        return nu_log

    def reset(self):
        pass
        # if hasattr(self, "A_log"):
        #     self.A_log.zero_()

    def forward(
        self, x: torch.Tensor, v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input data is (T B C H W)

        # Hidden state update
        # Sigmoid to prevent numerical instability
        T = x.size(0)
        Akernel = (-torch.exp(self.A_log)).unsqueeze(-1) * torch.arange(T).type_as(
            self.A_log
        ).repeat(*self.A_log.shape, 1)
        Akernel = torch.exp(Akernel).unsqueeze(0)

        x = x.permute(1, 2, 3, 4, 0).contiguous()  # (B C H W T)

        x_f = torch.fft.rfft(x, n=2 * T)
        Akernel_f = torch.fft.rfft(Akernel, n=2 * T)
        v = torch.fft.irfft(Akernel_f * x_f, n=2 * T)[..., :T]  # (B C H W T)

        v = v.permute(4, 0, 1, 2, 3).contiguous()
        s = spike(v - self.thresh)
        return s


class BLIF(nn.Module):  # BLIF do have reset
    def __init__(self, r_min, r_max, channels, thresh=1.0):
        super().__init__()

        # Shared between neurons in a layer
        self.init_flag = True
        self.r_min = r_min
        self.r_max = r_max
        # Just for convenience
        self.thresh = thresh
        A_log = self.init_recurrent(shape=[channels, 1, 1]).requires_grad_()
        self.register_parameter("A_log", nn.Parameter(A_log))

    def init_recurrent(self, data):
        u = torch.rand(size=data.size()[2:])
        nu_log = torch.log(
            -0.5 * torch.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2)
        )
        return nu_log

    def reset(self):
        del self.A_log

    def forward(
        self, x: torch.Tensor, v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input data is (T B C H W)

        # we initialize the parameter at the first time of input
        if isinstance(self.A_log, float):
            A_log = self.init_recurrent(x.data)
            self.A_log = nn.Parameter(A_log)  # (C H W)

        # Hidden state update
        # Sigmoid to prevent numerical instability
        T = x.size(0)
        Akernel = (-torch.exp(self.A_log)).unsqueeze(-1) * torch.arange(T).type_as(
            self.A_log
        ).repeat(*self.A_log.shape, 1)
        Akernel = torch.exp(Akernel).unsqueeze(0)

        # TODO(hujiakui): if flatten

        x = x.permute(1, 2, 3, 4, 0).contiguous()  # (B C H W T)
        Akernel = Akernel.unsqueeze(0)  # (1 C H W T)

        x_f = torch.fft.rfft(x, n=2 * T)
        Akernel_f = torch.fft.rfft(Akernel, n=2 * T)
        v = torch.fft.irfft(Akernel_f * x_f, n=2 * T)[..., :T]  # (B C H W T)

        v = v.permute(4, 0, 1, 2, 3).contiguous()
        s = spike(v - self.thresh)
        # TODO(): Warp
        s[1:, ...] = s[1:, ...] * (1.0 - s[: T - 1, ...])
        return s


class WarpSpikeFunction(Function):
    @staticmethod
    def forward(ctx, s):
        T = s.shape[0]
        s[1:, ...] = s[1:, ...] * (1.0 - s[: T - 1, ...])
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * result
