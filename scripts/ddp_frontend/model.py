"""DDP perception front end: 6 cameras -> Fused BEV, rebuilt to match last.ckpt.

Module names follow the checkpoint keys. Behaviour follows the authors' public
repository ChampagneAndfragrance/Anonomous_DDP. The target reaches perception as
raw ego-frame x m, y m, yaw deg; the /[10, 10, 180] scaling is planner-only.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet18


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim*4), nn.Linear(dim*4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x*self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = Attention(dim, heads), Mlp(dim)
        self.ls1, self.ls2 = LayerScale(dim), LayerScale(dim)

    def forward(self, x):
        x = x+self.ls1(self.attn(self.norm1(x)))
        return x+self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV2(nn.Module):
    """dinov2_vitb14_reg with the position handling of the official hub entry."""

    def __init__(self, dim=768, depth=12, heads=12, patch=14, registers=4, grid=37):
        super().__init__()
        self.patch, self.registers = patch, registers
        self.patch_embed = PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid*grid+1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def position(self, height, width):
        cls, grid = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        side = int(math.sqrt(grid.shape[1]))
        grid = grid.reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
        grid = F.interpolate(grid, size=(height, width), mode='bicubic', antialias=True)
        return torch.cat((cls, grid.permute(0, 2, 3, 1).flatten(1, 2)), 1)

    def forward(self, x):
        """Normalised patch tokens of the last block, as [B, C, H, W]."""
        height, width = x.shape[-2]//self.patch, x.shape[-1]//self.patch
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), 1)+self.position(height, width)
        x = torch.cat((x[:, :1], self.register_tokens.expand(len(x), -1, -1), x[:, 1:]), 1)
        for block in self.blocks:
            x = block(x)
        patches = self.norm(x)[:, 1+self.registers:]
        return patches.transpose(1, 2).reshape(len(x), -1, height, width)


class AsppConv(nn.Sequential):
    def __init__(self, channels, out, dilation):
        super().__init__(nn.Conv2d(channels, out, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out), nn.ReLU())


class AsppPooling(nn.Sequential):
    def __init__(self, channels, out):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, out, 1, bias=False),
                         nn.BatchNorm2d(out), nn.ReLU())

    def forward(self, x):
        return F.interpolate(super().forward(x), size=x.shape[-2:], mode='bilinear', align_corners=False)


class Aspp(nn.Module):
    def __init__(self, channels, out):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU()),
            *(AsppConv(channels, out, rate) for rate in (12, 24, 36)), AsppPooling(channels, out)])
        self.project = nn.Sequential(nn.Conv2d(out*5, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(),
                                     nn.Dropout(.5))

    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], 1))


class DeepLabHead(nn.Sequential):
    def __init__(self, channels, out, hidden=64):
        super().__init__(Aspp(channels, hidden), nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                         nn.BatchNorm2d(hidden), nn.ReLU(), nn.Conv2d(hidden, out, 1))


class Merge(nn.Module):
    """UpsamplingConcat at scale 1: every DINO level already has the 21x21 grid."""

    def __init__(self, channels, out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True))

    def forward(self, head, skip):
        return self.conv(torch.cat((skip, head), 1))


class CamEncoder(nn.Module):
    def __init__(self, cfg, depth_bins, dim=768):
        super().__init__()
        self.backbone = DinoV2(dim)
        self.depth_layer_1, self.feature_layer_1 = DeepLabHead(dim, dim), DeepLabHead(dim, dim)
        self.depth_layer_2 = Merge(dim*2, depth_bins)
        self.feature_layer_2 = Merge(dim*2, cfg['bev_encoder_in_channel'])

    def forward(self, images):
        tokens = self.backbone(images)
        return (self.feature_layer_2(self.feature_layer_1(tokens), tokens),
                self.depth_layer_2(self.depth_layer_1(tokens), tokens))


class BevModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bounds = torch.tensor([cfg['bev_x_bound'], cfg['bev_y_bound'], cfg['bev_z_bound']])
        self.register_buffer('bev_res', bounds[:, 2].clone())
        self.register_buffer('bev_start_pos', bounds[:, 0]+bounds[:, 2]/2)
        self.register_buffer('bev_dim', ((bounds[:, 1]-bounds[:, 0])/bounds[:, 2]).long())
        self.register_buffer('frustum', self.make_frustum(cfg))
        self.cam_encoder = CamEncoder(cfg, self.frustum.shape[0])

    @staticmethod
    def make_frustum(cfg):
        height, width = cfg['final_dim']
        rows, cols = height//cfg['bev_down_sample'], width//cfg['bev_down_sample']
        depth = torch.arange(*cfg['d_bound'], dtype=torch.float).view(-1, 1, 1).expand(-1, rows, cols)
        u = torch.linspace(0, width-1, cols).view(1, 1, cols).expand_as(depth)
        v = torch.linspace(0, height-1, rows).view(1, rows, 1).expand_as(depth)
        return torch.stack((u, v, depth), -1)

    def geometry(self, intrinsics, extrinsics):
        cam_to_ego = torch.inverse(extrinsics)
        rotation, translation = cam_to_ego[..., :3, :3], cam_to_ego[..., :3, 3]
        batch, cameras = translation.shape[:2]
        points = self.frustum[None, None, ..., None]
        points = torch.cat((points[..., :2, :]*points[..., 2:3, :], points[..., 2:3, :]), -2)
        transform = rotation.matmul(torch.inverse(intrinsics)).view(batch, cameras, 1, 1, 1, 3, 3)
        return transform.matmul(points).squeeze(-1)+translation.view(batch, cameras, 1, 1, 1, 3)

    def splat(self, geometry, features):
        """Sum camera features into BEV cells. Output axes are [channel, x, y]."""
        batch, channels = len(features), features.shape[-1]
        dim_x, dim_y, dim_z = self.bev_dim.tolist()
        output = features.new_zeros(batch, dim_x*dim_y, channels)
        cells = ((geometry-(self.bev_start_pos-self.bev_res/2))/self.bev_res).long()
        for b in range(batch):
            cell = cells[b].reshape(-1, 3)
            mask = ((cell >= 0) & (cell < self.bev_dim)).all(-1)
            output[b].index_add_(0, cell[mask, 0]*dim_y+cell[mask, 1], features[b].reshape(-1, channels)[mask])
        return output.transpose(1, 2).reshape(batch, channels, dim_x, dim_y)

    def forward(self, images, intrinsics, extrinsics):
        batch, cameras = images.shape[:2]
        features, depth = self.cam_encoder(images.flatten(0, 1))
        depth = depth.softmax(1)
        lifted = depth[:, None]*features[:, :, None]
        lifted = lifted.view(batch, cameras, *lifted.shape[1:]).permute(0, 1, 3, 4, 5, 2)
        return self.splat(self.geometry(intrinsics, extrinsics), lifted), depth


class BevEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)
        self.bn1, self.layer1, self.layer2 = trunk.bn1, trunk.layer1, trunk.layer2
        self.layer3, self.layer4 = trunk.layer3, trunk.layer4   # layer4 is stored but unused

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        return self.layer3(self.layer2(self.layer1(x))).flatten(2)


def state_encoder(inputs, tokens):
    hidden = tokens//2
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                         nn.ReLU(inplace=True), nn.Linear(hidden, tokens), nn.ReLU(inplace=True))


class FeatureFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, tokens = cfg['tf_en_dim'], cfg['tf_en_bev_length']
        layer = nn.TransformerEncoderLayer(dim, cfg['tf_en_heads'])
        self.tf_encoder = nn.TransformerEncoder(layer, cfg['tf_en_layers'], enable_nested_tensor=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens, dim))
        self.motion_encoder = state_encoder(cfg['tf_en_motion_length'], tokens)   # stored but unused
        self.target_encoder = state_encoder(cfg['tf_en_motion_length'], tokens)

    def forward(self, bev, target):
        target = self.target_encoder(target[:, None]).transpose(1, 2).expand(-1, -1, 8)
        fused = torch.cat((bev.transpose(1, 2), target), 2)+self.pos_embed
        return self.tf_encoder(fused.transpose(0, 1)).transpose(0, 1)


class FilmModulate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(3, dim*2), nn.ReLU())

    def forward(self, tokens, target):
        gamma, beta = self.film(target)[:, None].chunk(2, -1)
        return gamma*tokens+beta


class SegmentationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, width = cfg['bev_encoder_out_channel'], cfg['bev_encoder_in_channel']
        self.c5_conv = nn.Conv2d(channels, width, 1)
        self.up_conv5, self.up_conv4, self.up_conv3 = (nn.Conv2d(width, width, 1) for _ in range(3))
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, cfg['seg_classes'], 1))

    def forward(self, tokens):
        up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        side = int(math.sqrt(tokens.shape[1]))
        x = F.relu(self.c5_conv(tokens.transpose(1, 2).reshape(len(tokens), -1, side, side)))
        for conv in (self.up_conv5, self.up_conv4, self.up_conv3):
            x = F.relu(conv(up(x)))
        x = F.interpolate(x, size=(200, 200), mode='bilinear', align_corners=False)
        return self.segmentation_head(x)


class DdpFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bev_model = BevModel(cfg)
        self.bev_encoder = BevEncoder(cfg['bev_encoder_in_channel'])
        self.feature_fusion = FeatureFusion(cfg)
        self.film_modulate = FilmModulate(cfg['tf_en_dim'])
        self.segmentation_head = SegmentationHead(cfg)

    @classmethod
    def load(cls, path, device='cuda'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(saved['config'])
        model.load_state_dict(saved['state_dict'], strict=True)
        return model.to(device).eval().requires_grad_(False)

    def forward(self, images, intrinsics, extrinsics, target):
        """images [B,6,3,294,294], target [B,3] in ego x m, y m, yaw deg.

        `fused_bev` is the 267-channel planner input: 264 fused + 3 segmentation logits.
        """
        bev, depth = self.bev_model(images, intrinsics, extrinsics)
        tokens = self.film_modulate(self.feature_fusion(self.bev_encoder(bev), target), target)
        segmentation = self.segmentation_head(tokens)
        side = int(math.sqrt(tokens.shape[1]))
        fused = tokens.transpose(1, 2).reshape(len(tokens), -1, side, side)
        small = F.interpolate(segmentation, size=(side, side), mode='bilinear', align_corners=False)
        return dict(fused_bev=torch.cat((fused, small), 1), segmentation=segmentation, bev=bev, depth=depth)
