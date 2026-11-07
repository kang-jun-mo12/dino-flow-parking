"""Numeric scene tokens of k4_flow_solver, rebuilt from what the front end predicts.

Rows are [x, y, sin yaw, cos yaw, length, width, one-hot kind], in the ego frame of the
window start: vehicle rectangles from the predicted mask, target slot edges and the
+-10 m boundary. The lot polygon and K4 path of the old input are not in this data.
"""
from pathlib import Path
import numpy as np
import torch
from ddp_frontend import DEFAULT_WEIGHTS
from ddp_frontend.model import DdpFrontend
from k4_flow_solver.conditions import occupancy_rectangles

SLOT_LENGTH, SLOT_WIDTH = 5.5, 3.1
RECTANGLE, SLOT, BOUNDARY = 0, 2, 3


def row(xy, yaw, length, width, kind, cfg):
    scale = cfg.coordinate_scale
    return [xy[0]/scale, xy[1]/scale, np.sin(yaw), np.cos(yaw), length/scale, width/scale,
            *(float(i == kind) for i in range(4))]


def edges(polygon, kind, cfg):
    rows = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b-a
        rows.append(row((a+b)/2, np.arctan2(edge[1], edge[0]), np.linalg.norm(edge), 0., kind, cfg))
    return rows


def scene_rows(mask, target, cfg):
    """`mask` is the 200x200 vehicle class, indexed row=100+10x, col=100+10y."""
    x, y = target[0]*cfg.coordinate_scale, target[1]*cfg.coordinate_scale
    yaw = np.arctan2(target[2], target[3])
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])*[SLOT_LENGTH/2, SLOT_WIDTH/2]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    fixed = edges(corners@rotation.T+[x, y], SLOT, cfg)
    fixed += edges(np.array([[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]), BOUNDARY, cfg)
    rectangles = sorted(occupancy_rectangles(mask), key=lambda r: -(r[1]-r[0])*(r[3]-r[2]))
    rows = [row((((r0+r1)/2-100)/10, ((c0+c1)/2-100)/10), 0., (r1-r0)/10, (c1-c0)/10, RECTANGLE, cfg)
            for r0, r1, c0, c1 in rectangles[:cfg.scene_tokens-len(fixed)]]
    return fixed+rows, len(rectangles)


def build_tokens(folder, cfg):
    path = Path(folder)/'tokens.npz'
    if path.exists():
        return path
    bev = np.load(Path(folder)/'bev.npy', mmap_mode='r')
    with np.load(Path(folder)/'labels.npz') as z:
        targets = z['target']
    head = DdpFrontend.load(DEFAULT_WEIGHTS).eval().segmentation_head
    tokens = np.zeros((len(bev), cfg.scene_tokens, 10), np.float16)
    valid = np.zeros((len(bev), cfg.scene_tokens), bool)
    counts = []
    for start in range(0, len(bev), 32):
        rows = torch.as_tensor(np.asarray(bev[start:start+32])).cuda().float()
        with torch.no_grad():
            masks = head(rows[:, :264].flatten(2).transpose(1, 2)).argmax(1).cpu().numpy() == 1
        for offset, mask in enumerate(masks):
            values, count = scene_rows(mask, targets[start+offset], cfg)
            tokens[start+offset, :len(values)] = values
            valid[start+offset, :len(values)] = True
            counts.append(count)
    np.savez(path, tokens=tokens, valid=valid)
    print(f'scene tokens: rectangles median {np.median(counts):.0f}, max {max(counts)}, '
          f'cut windows {np.mean(np.array(counts) > cfg.scene_tokens-8)*100:.2f}%', flush=True)
    return path
