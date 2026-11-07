"""Rebuild the 200x200 segmentation of every packed window from its cached Fused BEV.

Usage: python scripts/structure_flow/make_seg.py PACK

The segmentation head only reads the 264 fused channels, so no camera image is needed.
Writes PACK/seg.npy, uint8 class ids [N, 200, 200].
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddp_frontend import DEFAULT_WEIGHTS
from ddp_frontend.model import DdpFrontend


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('--frontend', default=DEFAULT_WEIGHTS)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    head = DdpFrontend.load(args.frontend).segmentation_head
    bev = np.load(args.pack/'bev.npy', mmap_mode='r')
    seg = np.lib.format.open_memmap(args.pack/'seg.npy', 'w+', np.uint8, (len(bev), 200, 200))
    worst = 0.
    for start in range(0, len(bev), args.batch):
        part = torch.as_tensor(np.asarray(bev[start:start+args.batch])).cuda().float()
        logits = head(part[:, :264].flatten(2).transpose(1, 2))
        small = F.interpolate(logits, size=part.shape[-2:], mode='bilinear', align_corners=False)
        worst = max(worst, (small-part[:, 264:]).abs().max().item())
        seg[start:start+len(part)] = logits.argmax(1).to(torch.uint8).cpu().numpy()
    seg.flush()
    share = np.bincount(np.asarray(seg[::97]).ravel(), minlength=3)/seg[::97].size
    print(f'windows {len(seg)}, largest gap to the cached logits {worst:.4f}, class share {np.round(share, 3).tolist()}')


if __name__ == '__main__':
    main()
