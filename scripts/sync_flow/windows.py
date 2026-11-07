"""E2E-format parking tasks -> ego-frame 16-waypoint supervision windows.

A task folder holds measurements/%04d.json and parking_goal/0001.json. Poses use
CARLA's frame unchanged; the planar rotation below does not depend on handedness.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .config import DEFAULT


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def to_ego(pose, points):
    """Express world [...,3] poses (x, y, yaw rad) in the frame of `pose`."""
    points = np.asarray(points, np.float64)
    dx, dy = points[..., 0]-pose[0], points[..., 1]-pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((c*dx+s*dy, -s*dx+c*dy, wrap(points[..., 2]-pose[2])), -1)


def to_world(pose, points):
    points = np.asarray(points, np.float64)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((pose[0]+c*points[..., 0]-s*points[..., 1],
                     pose[1]+s*points[..., 0]+c*points[..., 1],
                     wrap(points[..., 2]+pose[2])), -1)


@dataclass
class Task:
    path: Path
    pose: np.ndarray      # [T,3] world x, y, yaw rad
    speed: np.ndarray     # [T] signed m/s
    steer: np.ndarray     # [T] CARLA command in [-1,1]
    reverse: np.ndarray   # [T] bool
    target: np.ndarray    # [3] world x, y, yaw rad

    def __len__(self):
        return len(self.speed)


def load_task(path):
    path = Path(path)
    cache = path/'sync_flow_arrays.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return Task(path, z['pose'], z['speed'], z['steer'], z['reverse'], z['target'])
    rows = []
    for name in sorted((path/'measurements').glob('*.json')):
        rows.append(json.loads(name.read_text(encoding='utf-8')))
    if not rows:
        raise ValueError(f'no measurements in {path}')
    goal = json.loads((path/'parking_goal'/'0001.json').read_text(encoding='utf-8'))
    reverse = np.array([bool(r['Reverse']) for r in rows])
    # Measurements store unsigned km/h; the reverse flag carries the direction.
    speed = np.array([r['speed'] for r in rows], np.float64)/3.6*np.where(reverse, -1., 1.)
    pose = np.array([[r['x'], r['y'], np.deg2rad(r['yaw'])] for r in rows], np.float64)
    steer = np.array([r['Steer'] for r in rows], np.float64)
    target = np.array([goal['x'], goal['y'], np.deg2rad(goal['yaw'])], np.float64)
    np.savez(cache, pose=pose, speed=speed, steer=steer, reverse=reverse, target=target)
    return Task(path, pose, speed, steer, reverse, target)


def target_features(target_ego, cfg=DEFAULT):
    x, y, yaw = target_ego
    return np.array([x/cfg.coordinate_scale, y/cfg.coordinate_scale,
                     np.sin(yaw), np.cos(yaw)], np.float32)


def ego_features(speed, steer, reverse, cfg=DEFAULT):
    return np.array([speed/cfg.max_speed_mps, steer, float(reverse)], np.float32)


def window(task, frame, cfg=DEFAULT):
    """Future waypoints from `frame`. Past the task end the vehicle holds at rest,
    which is what teaches the planner to stop on the goal."""
    last = len(task)-1
    wanted = frame+cfg.stride_frames*np.arange(1, cfg.horizon+1)
    index = np.minimum(wanted, last)
    origin = task.pose[frame]
    relative = to_ego(origin, task.pose[index])
    # Unwrapped so the yaw channel stays continuous inside a chunk.
    relative[:, 2] = np.unwrap(np.r_[0., relative[:, 2]])[1:]
    speed = np.where(wanted <= last, task.speed[index], 0.)
    chunk = np.c_[relative, speed].astype(np.float32)
    return dict(chunk=chunk, valid=wanted <= last,
                target=target_features(to_ego(origin, task.target), cfg),
                ego=ego_features(task.speed[frame], task.steer[frame], task.reverse[frame], cfg))
