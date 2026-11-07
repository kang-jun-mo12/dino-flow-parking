"""Relabel a packed dataset so that every expert path ends at a chosen depth in the slot.

Usage: python scripts/structure_flow/trim_labels.py PACK OUT [--stop-depth -0.30] [--mode squeeze]

The planner reads the goal from the Fused BEV, which is cached, so only the path labels can
change. The evaluator's dead reckoning puts the goal about 0.3 m too deep once the car has
turned 90 degrees, and experts stop about 0.18 m past the slot centre with some spread. Ending
each path at one depth removes the spread and offsets the drift.

`squeeze` moves the end of the path up and shrinks the nearly straight depth before it, so it
keeps the heading experts fix in the last half metre. Scaling the whole path broke the curvature
limit in 23% of the windows. `cut` stops the path at that depth and loses the heading fix: label
end heading error goes from 1 to 4 degrees.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_flow.config import DEFAULT as cfg
from structure_flow.labels import rear_axle, window_label
from sync_flow.windows import load_task, to_ego, to_world


def trimmed(task, stop_depth):
    """The task cut where the car centre first reaches `stop_depth` in its last reverse run."""
    inward = 1. if task.target[0] > 285.6 else -1.
    depth = (task.pose[:, 0]-task.target[0])*inward
    forward = np.flatnonzero(~task.reverse)
    first = forward[-1]+1 if len(forward) else 0
    reached = np.flatnonzero(depth[first:] >= stop_depth)
    if not len(reached) or first+reached[0] >= len(task)-1:
        return task, len(task)-1
    cut = first+int(reached[0])
    rear = rear_axle(task.pose, cfg)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(rear[:, :2], axis=0), axis=1))]
    left_before, left_now = arc[-1]-arc, arc[cut]-arc[:cut+1]
    speed = task.speed[:cut+1].copy()
    run = slice(first, cut+1)
    # left_before falls along the run, so reverse it for interp.
    tail = slice(first, None)
    speed[run] = np.sign(task.speed[run])*np.interp(left_now[run], left_before[tail][::-1], np.abs(task.speed[tail])[::-1])
    return replace(task, pose=task.pose[:cut+1], speed=speed, steer=task.steer[:cut+1], reverse=task.reverse[:cut+1]), cut


def squeezed(task, frame, stop_depth, keep=.7, least=.05):
    """The task with the path from `frame` ending at `stop_depth`, or None past that depth.

    The last `keep` metres hold the heading fix and move as they are, at the curvature the expert
    drove. Only the depth before them shrinks, where the path is nearly straight. A car already
    inside the moved part takes the expert path from the depth it would have there.
    """
    inward = 1. if task.target[0] > 285.6 else -1.
    depth = (task.pose[:, 0]-task.target[0])*inward
    forward = np.flatnonzero(~task.reverse)
    first = forward[-1]+1 if len(forward) else 0
    anchor = max(frame, first)
    shift = depth[-1]-stop_depth
    if anchor >= len(task)-1 or depth[anchor] >= stop_depth or shift <= 0:
        return None
    fix = first+int(np.flatnonzero(depth[first:] >= depth[-1]-keep)[0])
    room = depth[fix]-depth[anchor]
    if room-shift >= least:
        pose = task.pose.copy()
        pose[anchor:fix, 0] = pose[anchor, 0]+(room-shift)/room*(pose[anchor:fix, 0]-pose[anchor, 0])
        pose[fix:, 0] -= shift*inward
        return replace(task, pose=pose)
    ahead = np.flatnonzero(depth[anchor:] >= depth[anchor]+shift)
    if not len(ahead) or anchor+ahead[0] >= len(task)-1:
        return None
    match = anchor+int(ahead[0])
    moved = to_world(task.pose[anchor], to_ego(task.pose[match], task.pose[match:]))
    join = lambda values, tail: np.concatenate((values[:anchor], tail))
    return replace(task, pose=join(task.pose, moved), speed=join(task.speed, task.speed[match:]),
                   steer=join(task.steer, task.steer[match:]), reverse=join(task.reverse, task.reverse[match:]))


def reaching(task, frame, near=-3., longest=112):
    """Window settings that reach the expert's stop from near the slot.

    Experts creep over the last half metre, so 17% of the windows there ran out of time short of
    the stop, at scattered depths, and taught plans that end anywhere.
    """
    inward = 1. if task.target[0] > 285.6 else -1.
    moving = np.flatnonzero(np.abs(task.speed) > .02)
    left = (moving[-1] if len(moving) else len(task)-1)-frame
    if not task.reverse[frame] or (task.pose[frame, 0]-task.target[0])*inward < near or left <= cfg.window_frames:
        return cfg
    return replace(cfg, horizon=-(-min(left, longest)//cfg.stride_frames))


def task_labels(path, stop_depth, mode):
    task = load_task(path)
    short, cut = trimmed(task, stop_depth)
    rows, stops = [], 0
    for frame in range(0, len(task), cfg.stride_frames):
        if mode == 'cut':
            label = window_label(short, min(frame, cut), cfg)
        else:
            changed = squeezed(task, frame, stop_depth)
            stops += changed is None
            # Too close to shift: what is left of the path, cut at the same depth.
            label = window_label(changed, frame, reaching(changed, frame)) if changed else window_label(short, min(frame, cut), cfg)
        truth = window_label(task, frame, cfg)
        rows.append(dict(label, target=truth['target'], ego=truth['ego']))
    return rows, task.speed[::cfg.stride_frames], stops if mode != 'cut' else len(task)-1-cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--roots', nargs='+', type=Path, required=True, help='cached e2e_dataset and parkinglot_1000 folders')
    parser.add_argument('--stop-depth', type=float, default=-.3)
    parser.add_argument('--mode', choices=('squeeze', 'cut'), default='squeeze')
    args = parser.parse_args()
    paths = sorted(p.parent.parent for root in args.roots for p in root.glob('*/task*/fused_bev/summary.json'))
    with np.load(args.pack/'labels.npz') as z:
        old = {key: z[key] for key in z.files}
    with ThreadPoolExecutor(8) as pool:
        tasks = list(pool.map(lambda p: task_labels(p, args.stop_depth, args.mode), paths))
    rows = [row for labels, _, _ in tasks for row in labels]
    speed = np.concatenate([value for _, value, _ in tasks]).astype(np.float32)
    cuts = np.array([value for _, _, value in tasks])
    target, ego = np.stack([r['target'] for r in rows]), np.stack([r['ego'] for r in rows])
    # Same windows in the same order as the pack, or the cached BEV would not match.
    assert len(rows) == len(old['h']), (len(rows), len(old['h']))
    assert np.abs(target-old['target']).max() < 1e-5 and np.abs(ego-old['ego']).max() < 1e-5
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out/'bev.npy').exists():
        os.link(args.pack/'bev.npy', args.out/'bev.npy')
    np.savez(args.out/'labels.npz', h=np.array([r['h'] for r in rows]), latent=np.stack([r['latent'] for r in rows]),
             target=target, ego=ego, speed=speed, val=old['val'])
    (args.out/'selection.json').write_text((args.pack/'selection.json').read_text())
    changed = (np.array([r['h'] for r in rows]) != old['h']).mean()
    print(json.dumps(dict(tasks=len(paths), windows=len(rows), stop_depth=args.stop_depth, mode=args.mode, per_task_mean=float(cuts.mean()),
                          per_task_max=int(cuts.max()), structure_changed=float(changed))))


if __name__ == '__main__':
    main()
