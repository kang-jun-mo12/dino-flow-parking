"""Train StructureFlow on cached Fused BEV windows.

Usage: python scripts/structure_flow/train.py ROOT [ROOT ...] [--out DIR] [--epochs N]

Windows are packed once into a memory-mapped array, since the cache is one file per
frame. Tasks are split whole, so neighbouring frames never cross train and validation.
With --train-tasks the crowded goal slots are trimmed to reach that count, and the tasks
left out become the validation set. Without it, whole runs are held out.
Validation also repeats the Flow loss with the BEV of another scene: if that loss does
not rise, the model is not using the BEV.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zlib
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_flow.config import DEFAULT as cfg
from structure_flow.labels import decode, window_label
from structure_flow.model import StructureFlow
from structure_flow.tokens import build_tokens
from sync_flow.windows import load_task


def task_windows(path):
    task = load_task(path)
    frames = list(range(0, len(task), cfg.stride_frames))
    return path, frames, [window_label(task, frame, cfg) for frame in frames], task.speed[frames]


def goal_slot(path):
    goal = json.loads((path/'parking_goal'/'0001.json').read_text())
    return round(goal['x']), round(goal['y'])


def trim_slots(paths, count):
    """Validation tasks: the fewest per-slot cuts that leave `count` tasks, largest slots first."""
    if len(paths) <= count:
        sys.exit(f'{len(paths)} tasks, cannot keep {count} and still validate')
    slots = {}
    for path in sorted(paths, key=lambda p: zlib.crc32(str(p).encode())):
        slots.setdefault(goal_slot(path), []).append(path)
    held = []
    while len(held) < len(paths)-count:
        held.append(max(slots.values(), key=len).pop())
    return set(held), {f'{slot[0]},{slot[1]}': len(items) for slot, items in sorted(slots.items())}


def pack(roots, out, train_tasks=None):
    paths = sorted(p.parent.parent for root in roots for p in root.glob('*/task*/fused_bev/summary.json'))
    digest = hashlib.sha1(json.dumps([str(p) for p in paths]+[cfg.to_dict(), train_tasks]).encode()).hexdigest()[:12]
    folder = out/f'pack_{digest}'
    if (folder/'labels.npz').exists():
        return folder, len(paths)
    folder.mkdir(parents=True, exist_ok=True)
    held = None
    if train_tasks:
        held, kept = trim_slots(paths, train_tasks)
        (folder/'selection.json').write_text(json.dumps(dict(train_per_slot=kept, val=[str(p) for p in sorted(held)]), indent=1))
        print(f'train {train_tasks}, val {len(held)}, smallest slot {min(kept.values())}, '
              f'largest slot {max(kept.values())}', flush=True)
    with ThreadPoolExecutor(8) as pool:
        tasks = list(pool.map(task_windows, paths))
    count = sum(len(frames) for _, frames, _, _ in tasks)
    bev = np.lib.format.open_memmap(folder/'bev.npy', 'w+', np.float16,
                                    (count, cfg.bev_channels, cfg.token_grid, cfg.token_grid))
    rows, start = [], 0
    with ThreadPoolExecutor(8) as pool:
        for path, frames, labels, speed in tasks:
            files = [path/'fused_bev'/f'{frame:04d}.npy' for frame in frames]
            bev[start:start+len(frames)] = np.stack(list(pool.map(np.load, files)))
            start += len(frames)
            val = path in held if held is not None else zlib.crc32(path.parent.name.encode()) % 10 == 0
            rows += [(label, value, val) for label, value in zip(labels, speed)]
    bev.flush()
    np.savez(folder/'labels.npz', h=np.array([r[0]['h'] for r in rows]), latent=np.stack([r[0]['latent'] for r in rows]),
             target=np.stack([r[0]['target'] for r in rows]), ego=np.stack([r[0]['ego'] for r in rows]),
             speed=np.array([r[1] for r in rows], np.float32), val=np.array([r[2] for r in rows]))
    return folder, len(paths)


class Data:
    def __init__(self, folder, condition='bev'):
        self.bev = None if condition == 'numeric' else np.load(folder/('seg.npy' if condition == 'seg' else 'bev.npy'), mmap_mode='r')
        with np.load(folder/'labels.npz') as z:
            self.arrays = {key: z[key] for key in z.files}
        self.scene = None
        if condition in ('numeric', 'both'):
            with np.load(build_tokens(folder, cfg)) as z:
                self.scene, self.valid = z['tokens'], z['valid']

    def inputs(self, index, scene_index=None):
        """Encoder inputs. `scene_index` swaps the scene and keeps the target and ego state."""
        source = index if scene_index is None else scene_index
        bev = scene = valid = None
        if self.bev is not None:
            order = np.sort(source)
            bev = torch.as_tensor(np.asarray(self.bev[order])[np.searchsorted(order, source)]).cuda().float()
        if self.scene is not None:
            scene = torch.as_tensor(self.scene[source]).cuda().float()
            valid = torch.as_tensor(self.valid[source]).cuda()
        tensor = lambda key: torch.as_tensor(self.arrays[key][index]).cuda()
        return bev, tensor('target'), tensor('ego'), scene, valid

    def truth(self, index):
        return (torch.as_tensor(self.arrays['h'][index]).cuda().long(),
                torch.as_tensor(self.arrays['latent'][index]).cuda())


def normalization(arrays, train):
    latent, h = arrays['latent'][train], arrays['h'][train]
    used = np.arange(cfg.max_phases)[None] <= (h % cfg.max_phases)[:, None]
    lengths = latent[..., 0, 0][used]
    rest = latent[used][:, :, 1:].reshape(-1, 2)
    return dict(mean=[float(lengths.mean()), *map(float, rest.mean(0))],
                std=[float(lengths.std()+1e-6), *map(float, rest.std(0)+1e-6)])


@torch.no_grad()
def evaluate(model, data, index, weights):
    model.eval()
    generator = torch.Generator('cuda').manual_seed(0)
    partner = index[(np.arange(len(index))+len(index)//2) % len(index)]
    total = dict(ce=0., correct=0., flow=0., flow_other_bev=0.)
    for start in range(0, len(index), 256):
        part, other = index[start:start+256], partner[start:start+256]
        h, latent = data.truth(part)
        context, logits = model.encode(*data.inputs(part))
        noise = torch.randn(latent.shape, device='cuda', generator=generator)
        t = torch.rand(len(h), device='cuda', generator=generator)
        total['ce'] += F.cross_entropy(logits, h, weight=weights).item()*len(h)
        total['correct'] += (logits.argmax(-1) == h).sum().item()
        total['flow'] += model.loss(latent, context, h, noise, t).item()*len(h)
        total['flow_other_bev'] += model.loss(latent, model.encode(*data.inputs(part, other))[0], h, noise, t).item()*len(h)
    return {key: value/len(index) for key, value in total.items()}


@torch.no_grad()
def path_error(model, data, index, count=512):
    """End-point gap between a sampled path and the label, given the true structure."""
    model.eval()
    index = index[np.linspace(0, len(index)-1, min(count, len(index))).astype(int)]
    h, latent = data.truth(index)
    sampled = model.rollout(model.encode(*data.inputs(index))[0], h).cpu().numpy()
    gaps, speed = [], data.arrays['speed'][index]
    for value, label, structure, start in zip(sampled, latent.cpu().numpy(), h.tolist(), speed):
        gaps.append(np.linalg.norm(decode(value, structure, start, cfg)[-1, :2]-decode(label, structure, start, cfg)[-1, :2]))
    return float(np.median(gaps)), float(np.percentile(gaps, 95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='*', type=Path)
    parser.add_argument('--pack', type=Path, help='reuse a packed dataset, ignoring roots')
    parser.add_argument('--train-tasks', type=int, help='trim crowded goal slots to this many training tasks')
    parser.add_argument('--no-bev-attention', action='store_true')
    parser.add_argument('--condition', choices=('bev', 'seg', 'numeric', 'both'), default='bev')
    parser.add_argument('--out', type=Path, default=Path('runs'))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    folder, tasks = (args.pack, 0) if args.pack else pack(args.roots, args.out, args.train_tasks)
    network = replace(cfg, bev_attention=not args.no_bev_attention, condition=args.condition)
    data = Data(folder, args.condition)
    val = np.flatnonzero(data.arrays['val'])
    train = np.flatnonzero(~data.arrays['val'])
    counts = np.bincount(data.arrays['h'][train], minlength=2*cfg.max_phases).astype(np.float64)
    weights = np.where(counts > 0, 1/np.sqrt(np.maximum(counts, 1)), 0.)
    weights = torch.tensor(weights/weights[counts > 0].mean(), dtype=torch.float32, device='cuda')
    norm = normalization(data.arrays, train)
    print(f'tasks {tasks}, train {len(train)}, val {len(val)}, structures {counts.astype(int).tolist()}, '
          f'pack {time.time()-started:.0f} s', flush=True)
    model = StructureFlow(network, norm).cuda()
    print(f'parameters {sum(p.numel() for p in model.parameters())/1e6:.2f} M', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs*math.ceil(len(train)/args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=.05)
    best = math.inf
    for epoch in range(args.epochs):
        model.train()
        order, seen, ce_sum, flow_sum = rng.permutation(train), 0, 0., 0.
        for start in range(0, len(order), args.batch):
            part = order[start:start+args.batch]
            h, latent = data.truth(part)
            context, logits = model.encode(*data.inputs(part))
            ce = F.cross_entropy(logits, h, weight=weights)
            flow = model.loss(latent, context, h)
            optimizer.zero_grad(set_to_none=True)
            (ce+flow).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            seen += len(h)
            ce_sum += ce.item()*len(h)
            flow_sum += flow.item()*len(h)
        metrics = evaluate(model, data, val, weights)
        score = metrics['flow']+1-metrics['correct']  # val ce climbs from overconfidence alone, so rank by errors
        print(f"{epoch+1:3d} train ce {ce_sum/seen:.3f} flow {flow_sum/seen:.3f} | val ce {metrics['ce']:.3f} "
              f"acc {metrics['correct']:.3f} flow {metrics['flow']:.3f} other-BEV {metrics['flow_other_bev']:.3f} "
              f"| {time.time()-started:.0f} s", flush=True)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), config=network.to_dict(), normalization=norm,
                            epoch=epoch+1, metrics=metrics), args.out/'best.pt')
    state = torch.load(args.out/'best.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(state['state'])
    median, tail = path_error(model, data, val)
    print(f"best epoch {state['epoch']}: {state['metrics']}", flush=True)
    print(f'sampled path end gap m: median {median:.3f}, 95% {tail:.3f}', flush=True)


if __name__ == '__main__':
    main()
