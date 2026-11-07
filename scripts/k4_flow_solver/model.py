"""The two blocks of the phase U-Net that StructureFlow shares with the earlier whole-path solver."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def time_embedding(t, dimension):
    frequency = torch.exp(torch.linspace(0, math.log(1000), dimension//2, device=t.device))
    angle = t[:, None]*frequency[None]*2*math.pi
    return torch.cat((angle.sin(), angle.cos()), dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self, inputs, outputs, context):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, outputs, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, outputs)
        self.condition = nn.Linear(context, outputs*2)
        self.conv2 = nn.Conv1d(outputs, outputs, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, outputs)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, context):
        scale, shift = self.condition(context).chunk(2, dim=-1)
        value = self.norm1(self.conv1(x))*(1+scale[..., None])+shift[..., None]
        value = self.norm2(self.conv2(F.silu(value)))
        return F.silu(value+self.skip(x))
