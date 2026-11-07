"""Do sampled paths avoid the vehicles the front end sees, and does the scene input matter?

Usage: eval_structure_flow.py PACK CHECKPOINT [CHECKPOINT ...] [--windows 500] [--samples 4]

The vehicle mask comes from the DDP segmentation head on the cached BEV. Each path is
swept with the car body; the same sweep on the label paths calibrates the metric. Each
model also runs with the scene input of another window, keeping target and ego state.
"""
import argparse
from dataclasses import fields, replace
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ddp_frontend import DEFAULT_WEIGHTS
from ddp_frontend.model import DdpFrontend
from structure_flow.config import DEFAULT as cfg
from structure_flow.labels import decode, is_stop
from structure_flow.model import StructureFlow
from structure_flow.tokens import build_tokens

LENGTH, WIDTH = 4.4, 1.9
SHIFTS = (0, 125, 250)
U, V = (a.ravel() for a in np.meshgrid(np.linspace(-.5, .5, 9)*LENGTH, np.linspace(-.5, .5, 5)*WIDTH))


def hits(path, vehicles):
    """True if the body, swept along a rear-axle path, enters the vehicle mask."""
    pose = path[::8]
    yaw = pose[:, 2:3]
    # The window starts at the ego centre, cfg.rear_offset_m ahead of the rear axle.
    x = pose[:, :1]-cfg.rear_offset_m+cfg.rear_offset_m*np.cos(yaw)+np.cos(yaw)*U-np.sin(yaw)*V
    y = pose[:, 1:2]+cfg.rear_offset_m*np.sin(yaw)+np.sin(yaw)*U+np.cos(yaw)*V
    row, col = (100+10*x).astype(int), (100+10*y).astype(int)
    inside = (row >= 0) & (row < 200) & (col >= 0) & (col < 200)
    return bool(vehicles[row[inside], col[inside]].any())


def load(path):
    state = torch.load(path, map_location='cuda', weights_only=True)
    saved = state['config']
    known = {field.name for field in fields(cfg)}
    values = {key: value for key, value in saved.items() if key in known}
    network = replace(cfg, **{**values, 'bev_attention': saved.get('bev_attention', False)})
    model = StructureFlow(network, state['normalization']).cuda().eval()
    model.load_state_dict(state['state'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--windows', type=int, default=500)
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args()
    bev = np.load(args.pack/'bev.npy', mmap_mode='r')
    with np.load(args.pack/'labels.npz') as z:
        arrays = {key: z[key] for key in z.files}
    with np.load(build_tokens(args.pack, cfg)) as z:
        scene_all, valid_all = z['tokens'], z['valid']
    val = np.flatnonzero(arrays['val'])
    moving = [i for i in val if not is_stop(arrays['latent'][i], int(arrays['h'][i]), cfg)]
    index = np.array(moving)[np.linspace(0, len(moving)-1, min(args.windows, len(moving))).astype(int)]
    count = len(index)
    head = DdpFrontend.load(DEFAULT_WEIGHTS).eval().segmentation_head
    rows = torch.as_tensor(np.asarray(bev[index])).cuda().float()
    scene = torch.as_tensor(scene_all[index]).cuda().float()
    valid = torch.as_tensor(valid_all[index]).cuda()
    with torch.no_grad():
        masks = np.concatenate([head(rows[i:i+32, :264].flatten(2).transpose(1, 2)).argmax(1).cpu().numpy() == 1
                                for i in range(0, count, 32)])
    speed = arrays['speed'][index].astype(float)
    structure = arrays['h'][index].astype(int)
    labels = [decode(arrays['latent'][i], int(arrays['h'][i]), float(arrays['speed'][i]), cfg) for i in index]
    print(f'windows {count}, label paths touching vehicles '
          f'{np.mean([hits(p, m) for p, m in zip(labels, masks)])*100:.1f}%')
    h = torch.as_tensor(structure).cuda().long()
    target, ego = (torch.as_tensor(arrays[key][index]).cuda() for key in ('target', 'ego'))
    for path in args.checkpoints:
        model = load(path)
        line = []
        for shift in SHIFTS:
            source = (torch.arange(count, device='cuda')+shift) % count
            torch.manual_seed(0)
            touched, gaps = [], []
            with torch.no_grad():
                for _ in range(args.samples):
                    for start in range(0, count, 125):
                        part = slice(start, start+125)
                        pick = source[part]
                        context = model.encode(rows[pick], target[part], ego[part], scene[pick], valid[pick])[0]
                        for offset, value in enumerate(model.rollout(context, h[part]).cpu().numpy()):
                            k = start+offset
                            decoded = decode(value, int(structure[k]), speed[k], cfg)
                            touched.append(hits(decoded, masks[k]))
                            gaps.append(np.linalg.norm(decoded[-1, :2]-labels[k][-1, :2]))
            line.append(f'shift {shift}: touch {np.mean(touched)*100:.1f}% '
                        f'gap {np.median(gaps):.2f}/{np.percentile(gaps, 95):.2f}')
        print(f'{path.parent.name} [{model.cfg.condition}] '+' | '.join(line), flush=True)


if __name__ == '__main__':
    main()
