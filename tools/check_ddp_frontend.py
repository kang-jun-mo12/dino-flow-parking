"""Compare the rebuilt DDP front end with `topdown` on public frames.

Usage: check_ddp_frontend.py WEIGHTS TASK_DIR [TASK_DIR ...] [--image out.png]
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ddp_frontend import rig
from ddp_frontend.model import DdpFrontend

COLOURS = np.array([[30, 30, 30], [240, 180, 60], [80, 220, 80]], np.uint8)


def frames(task):
    goal = json.loads((task/'parking_goal'/'0001.json').read_text())
    for path in sorted((task/'measurements').glob('*.json')):
        name = path.stem
        if not all((task/camera/f'{name}.png').exists() for camera in rig.CAMERAS):
            continue
        measurement = json.loads(path.read_text())
        target = rig.target_in_ego(measurement, goal)
        images = torch.stack([rig.prepare_image(Image.open(task/camera/f'{name}.png')) for camera in rig.CAMERAS])
        yield images, target, rig.segmentation_labels(task/'topdown'/f'encoded_{name}.png', target.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('tasks', nargs='+', type=Path)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    model = DdpFrontend.load(args.weights)
    print('strict load ok,', sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    intrinsics, extrinsics = rig.intrinsics()[None].cuda(), rig.extrinsics()[None].cuda()
    rows, tiles = [], []
    with torch.no_grad():
        for task in args.tasks:
            for images, target, labels in frames(task):
                out = model(images[None].cuda(), intrinsics, extrinsics, target[None].cuda())
                prediction = out['segmentation'].argmax(1)[0].cpu().numpy()
                centre = np.nan
                if (prediction == 2).any() and (labels == 2).any():
                    centre = np.linalg.norm(np.argwhere(prediction == 2).mean(0)-np.argwhere(labels == 2).mean(0))*.1
                rows.append((rig.iou(prediction, labels, 1), rig.iou(prediction, labels, 2), centre))
                tiles.append(np.hstack((COLOURS[labels], np.full((200, 6, 3), 255, np.uint8), COLOURS[prediction])))
        shape = tuple(out['fused_bev'].shape)
        batch = images[None].cuda(), intrinsics, extrinsics, target[None].cuda()
        for _ in range(3):
            model(*batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model(*batch)
        torch.cuda.synchronize()
    rows = np.array(rows)
    print(f'frames {len(rows)} fused_bev {shape}')
    print(f'vehicle IoU mean {np.nanmean(rows[:, 0]):.3f} median {np.nanmedian(rows[:, 0]):.3f}')
    print(f'target IoU mean {np.nanmean(rows[:, 1]):.3f} median {np.nanmedian(rows[:, 1]):.3f}')
    print(f'target centre error median {np.nanmedian(rows[:, 2]):.2f} m')
    print(f'latency fp32 {(time.perf_counter()-start)/20*1000:.1f} ms per frame')
    if args.image:
        count = len(tiles)-len(tiles) % 3
        grid = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, count, 3)])
        Image.fromarray(grid).save(args.image)


if __name__ == '__main__':
    main()
