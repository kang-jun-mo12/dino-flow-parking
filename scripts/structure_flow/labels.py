"""Task frames -> (structure h, per-phase spline latent), and the way back to a path.

Latent [max_phases, coefficients, 3]: log length at knot 0, curvature/scale, speed/max.
Curvature, not steering: yaw is then linear in the spline, and the driven curvature is
not cut off by a nominal steering limit.
Paths are rear-axle poses in the frame of the rear axle at the window start.
"""
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline
from sync_flow.windows import target_features, to_ego
from .config import DEFAULT


@lru_cache(maxsize=4)
def basis(coefficients, samples):
    knots = np.r_[np.zeros(4), np.linspace(0, 1, coefficients-2)[1:-1], np.ones(4)]
    return BSpline(knots, np.eye(coefficients), 3)((np.arange(samples)+.5)/samples)


@lru_cache(maxsize=4)
def fit_matrix(coefficients, samples):
    return np.linalg.pinv(basis(coefficients, samples))


def rear_axle(pose, cfg=DEFAULT):
    pose = np.array(pose, np.float64)
    pose[..., 0] -= cfg.rear_offset_m*np.cos(pose[..., 2])
    pose[..., 1] -= cfg.rear_offset_m*np.sin(pose[..., 2])
    return pose


def split_phases(task, start, end, cfg=DEFAULT):
    """[(gear, first, last)] over frames start..end, cut where the moving gear changes."""
    phases, first, gear = [], start, 0
    for frame in range(start, end+1):
        if abs(task.speed[frame]) < cfg.move_speed_mps:
            continue
        current = -1 if task.reverse[frame] else 1
        if gear and current != gear:
            phases.append((gear, first, frame))
            first = frame
        gear = current
    return phases+[(gear, first, end)] if gear else []


def fit_phase(pose, speed, gear, cfg=DEFAULT):
    """Spline coefficients [K,3] of one phase from its rear-axle poses and speeds."""
    yaw = np.unwrap(pose[:, 2])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1))]
    total, result = arc[-1], np.zeros((cfg.coefficients, 3))
    result[0, 0] = np.log(max(total, cfg.min_length_m))
    if total < cfg.min_length_m:
        return result
    keep = np.r_[True, np.diff(arc) > 1e-6]
    at = (np.arange(cfg.samples)+.5)/cfg.samples*total
    heading = np.interp(at, arc[keep], yaw[keep])
    curvature = np.gradient(heading, at)*gear/cfg.curvature_scale
    matrix = fit_matrix(cfg.coefficients, cfg.samples)
    result[:, 1] = matrix@curvature
    result[:, 2] = np.clip(matrix@(np.interp(at, arc[keep], speed[keep])/cfg.max_speed_mps), 0, 1)
    return result


def window_label(task, frame, cfg=DEFAULT):
    """Label of the window at `frame`. Only the first two phases are kept."""
    end = min(frame+cfg.window_frames, len(task)-1)
    rear = rear_axle(task.pose, cfg)
    phases = split_phases(task, frame, end, cfg)[:cfg.max_phases]
    if not phases:      # stationary: the current gear with no length
        phases = [(-1 if task.reverse[frame] else 1, frame, frame)]
    h = (0 if phases[0][0] > 0 else cfg.max_phases)+len(phases)-1
    latent = np.zeros((cfg.max_phases, cfg.coefficients, 3))
    for index, (gear, first, last) in enumerate(phases):
        pose = rear[first:last+1]
        latent[index] = fit_phase(pose, np.abs(task.speed[first:last+1]), gear, cfg)
    return dict(h=h, latent=latent.astype(np.float32),
                target=target_features(to_ego(task.pose[frame], task.target), cfg),
                ego=np.array([task.speed[frame]/cfg.max_speed_mps, task.steer[frame],
                              float(task.reverse[frame])], np.float32))


def gears(h, cfg=DEFAULT):
    first = 1 if h < cfg.max_phases else -1
    return [first*(-1)**index for index in range(h % cfg.max_phases+1)]


def decode(latent, h, start_speed=None, cfg=DEFAULT):
    """Rear-axle path [N,5]: x, y, yaw, signed speed, phase. Known speeds are imposed:
    the current speed at the start and zero at a gear change."""
    grid = basis(cfg.coefficients, cfg.samples)
    state, rows, directions = np.zeros(3), [], gears(h, cfg)
    for index, gear in enumerate(directions):
        length = np.exp(latent[index, 0, 0])
        curvature = grid@latent[index, :, 1]*cfg.curvature_scale
        coefficients = latent[index, :, 2].astype(np.float64).copy()
        if index == 0 and start_speed is not None:
            coefficients[0] = abs(start_speed)/cfg.max_speed_mps
        if index > 0:
            coefficients[0] = 0.
        if index < len(directions)-1:
            coefficients[-1] = 0.
        speed = (grid@coefficients).clip(0, 1)*cfg.max_speed_mps
        step = length/cfg.samples
        for bend, value in zip(curvature, speed):
            turn = gear*bend*step
            middle = state[2]+turn/2
            state = state+np.array([gear*step*np.cos(middle), gear*step*np.sin(middle), turn])
            rows.append((*state, gear*value, index))
    return np.array(rows)


def is_stop(latent, h, cfg=DEFAULT):
    return sum(np.exp(latent[index, 0, 0]) for index in range(h % cfg.max_phases+1)) < cfg.stop_length_m
