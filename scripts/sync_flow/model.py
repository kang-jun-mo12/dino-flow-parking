"""Conditional Flow Matching over a 16-waypoint chunk, conditioned on a fused BEV.

Same transport as the whole-path Flow: Gaussian noise to data, x_t=(1-t)x0+t*x1,
Euler rollout. Only the target changed, from a whole control sequence to a chunk.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from k4_flow_solver.model import TemporalBlock, time_embedding
from .config import DEFAULT


@dataclass
class Context:
    tokens: torch.Tensor
    pooled: torch.Tensor

    def repeat(self, count):
        return Context(self.tokens.repeat_interleave(count, 0),
                       self.pooled.repeat_interleave(count, 0))


class BevTokens(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, g = cfg.context_dim, cfg.token_grid
        self.grid = g
        self.project = nn.Sequential(nn.Conv2d(cfg.bev_channels, d, 1), nn.GroupNorm(1, d),
                                     nn.SiLU(), nn.Conv2d(d, d, 3, padding=1))
        self.position = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        axis = (torch.arange(g)+.5)/g*2-1
        self.register_buffer('cells', torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2))

    def forward(self, bev):
        value = F.adaptive_avg_pool2d(self.project(bev), self.grid).flatten(2).transpose(1, 2)
        return self.norm(value+self.position(self.cells)[None])


class SyncChunkFlow(nn.Module):
    def __init__(self, cfg=DEFAULT, normalization=None):
        super().__init__()
        self.cfg = cfg
        d, w = cfg.context_dim, cfg.width
        self.bev = BevTokens(cfg)
        self.state = nn.Sequential(nn.Linear(cfg.target_dim+cfg.ego_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.queries = nn.Parameter(torch.randn(cfg.queries, d)*.02)
        self.read = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.read_norm = nn.LayerNorm(d)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.down1 = TemporalBlock(cfg.channels+1, w, d)
        self.down2 = TemporalBlock(w, w*2, d)
        self.middle = TemporalBlock(w*2, d, d)
        self.scene = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        block = nn.TransformerEncoderLayer(d, cfg.heads, d*3, dropout=0., batch_first=True,
                                           activation='gelu', norm_first=True)
        self.mix = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.up2 = TemporalBlock(d+w*2, w*2, d)
        self.up1 = TemporalBlock(w*3, w, d)
        self.head = nn.Conv1d(w, cfg.channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        shape = (cfg.horizon, cfg.channels)
        norm = normalization or dict(mean=torch.zeros(shape), std=torch.ones(shape))
        self.register_buffer('chunk_mean', torch.as_tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('chunk_std', torch.as_tensor(norm['std'], dtype=torch.float32))
        self.register_buffer('knot', torch.linspace(0, 1, cfg.horizon))

    def normalize(self, chunk):
        return (chunk-self.chunk_mean)/self.chunk_std

    def denormalize(self, y):
        return y*self.chunk_std+self.chunk_mean

    def encode(self, bev, target, ego):
        """Runs once per decision; every candidate and Euler step reuses it."""
        tokens = self.bev(bev)
        state = self.state(torch.cat((target, ego), -1))
        query = self.queries[None]+state[:, None]
        query = query+self.read(query, tokens, tokens, need_weights=False)[0]
        return Context(tokens, self.read_norm(query).mean(1)+state)

    def forward(self, y, t, context):
        condition = context.pooled+self.time(time_embedding(t, self.cfg.context_dim))
        knot = self.knot[None, None].expand(len(y), 1, -1)
        a = self.down1(torch.cat((y.transpose(1, 2), knot), 1), condition)
        b = self.down2(F.avg_pool1d(a, 2), condition)
        c = self.middle(F.avg_pool1d(b, 2), condition).transpose(1, 2)
        c = c+time_embedding(torch.linspace(0, 1, c.shape[1], device=y.device), self.cfg.context_dim)[None]
        c = c+self.scene(c, context.tokens, context.tokens, need_weights=False)[0]
        c = self.mix(c).transpose(1, 2)
        d = self.up2(torch.cat((F.interpolate(c, size=b.shape[-1], mode='linear', align_corners=True), b), 1), condition)
        e = self.up1(torch.cat((F.interpolate(d, size=a.shape[-1], mode='linear', align_corners=True), a), 1), condition)
        return self.head(e).transpose(1, 2)

    def loss(self, bev, target, ego, chunk):
        data = self.normalize(chunk)
        noise = torch.randn_like(data)
        t = torch.rand(len(data), device=data.device)
        mixed = (1-t[:, None, None])*noise+t[:, None, None]*data
        return F.mse_loss(self(mixed, t, self.encode(bev, target, ego)), data-noise)

    @torch.no_grad()
    def sample(self, bev, target, ego, candidates=None, steps=None, generator=None):
        """Return physical chunks [batch, candidates, horizon, channels]."""
        cfg = self.cfg
        count, steps = candidates or cfg.candidates, steps or cfg.euler_steps
        context = self.encode(bev, target, ego).repeat(count)
        y = torch.randn((len(bev)*count, cfg.horizon, cfg.channels), device=bev.device,
                        dtype=bev.dtype, generator=generator)
        for k in range(steps):
            t = torch.full((len(y),), k/steps, device=y.device, dtype=y.dtype)
            y = y+self(y, t, context)/steps
        return self.denormalize(y).reshape(len(bev), count, cfg.horizon, cfg.channels)
