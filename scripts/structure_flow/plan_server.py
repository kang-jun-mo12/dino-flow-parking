"""Planner process for the closed-loop benchmark: 6 images + ego state -> a 16-step path.

The CARLA 0.9.11 client only exists for Python 3.7, so the models live here under the
worker runtime. Messages are one JSON header and raw image bytes, which keeps numpy
pickles out of the link between the two Python versions.
Candidates: the most likely structure first, its medoid among the paths whose current gear
stays off the vehicle mask. The same is tried on the next structure if it is likely enough,
then the plan is a stop.
"""
import argparse
import json
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tools')]
from ddp_frontend import rig
from ddp_frontend import DEFAULT_WEIGHTS
from ddp_frontend.model import DdpFrontend
from eval_structure_flow import hits, load
from structure_flow.labels import decode, is_stop
from sync_flow.windows import target_features

AUTHKEY = b'structure-flow-benchmark'
SHAPE = (len(rig.CAMERAS), rig.HEIGHT, rig.WIDTH, 3)
MIN_STRUCTURE = .05


class Planner:
    def __init__(self, checkpoint, frontend):
        self.frontend = DdpFrontend.load(frontend).eval()
        self.model = load(checkpoint)
        self.cfg = self.model.cfg
        self.intrinsics = rig.intrinsics().cuda()[None]
        self.extrinsics = rig.extrinsics().cuda()[None]

    @torch.no_grad()
    def perceive(self, images, state):
        pixels = torch.stack([rig.prepare_image(image) for image in images])[None].cuda()
        target = torch.tensor(state['target'], dtype=torch.float32)[None].cuda()
        out = self.frontend(pixels, self.intrinsics, self.extrinsics, target)
        # The cache holds fp16, so the planner sees the same rounding here.
        return out['fused_bev'].half().float(), out['segmentation'][0].argmax(0).cpu().numpy()

    @torch.no_grad()
    def plan(self, images, state):
        started = time.perf_counter()
        cfg = self.cfg
        bev, seg = self.perceive(images, state)
        x, y, yaw = state['target']
        target = target_features((x, y, np.radians(yaw)), cfg)
        ego = np.array([state['speed']/cfg.max_speed_mps, state['steer'], float(state['reverse'])], np.float32)
        scene = torch.as_tensor(seg)[None].cuda() if cfg.condition == 'seg' else bev
        latents, structures, probabilities = self.model.sample(
            scene, torch.as_tensor(target)[None].cuda(), torch.as_tensor(ego)[None].cuda())
        latents, structures = latents.cpu().numpy(), structures.cpu().numpy()
        vehicles = seg == 1
        chosen, stop, safe_counts = None, False, []
        for h in dict.fromkeys(structures.tolist()):
            if probabilities[h] < MIN_STRUCTURE:    # the runner-up can be a structure with no training data
                continue
            group = [latent for latent, value in zip(latents, structures) if value == h]
            paths = [decode(latent, h, state['speed'], cfg) for latent in group]
            if all(is_stop(latent, h, cfg) for latent in group):
                chosen, stop = (h, paths[0]), True
                safe_counts.append(len(paths))
                break
            # Only the current gear: the next phase is planned again before the car gets there.
            safe = [path for path in paths if not hits(path[path[:, 4] == 0], vehicles)]
            safe_counts.append(len(safe))
            if safe:
                ends = np.array([path[-1, :2] for path in safe])
                centre = np.linalg.norm(ends[:, None]-ends[None], axis=-1).sum(1).argmin()
                chosen = (h, safe[centre])
                break
        torch.cuda.synchronize()
        reply = dict(stop=stop or chosen is None, blocked=chosen is None, safe=safe_counts,
                     probabilities=probabilities.cpu().tolist(), seconds=time.perf_counter()-started,
                     mask=target_mask_centre(seg))
        if chosen is not None:
            reply.update(h=int(chosen[0]), path=np.round(chosen[1], 4).tolist())
        return reply, seg


def target_mask_centre(seg):
    """Centre of the target class in the vehicle frame, as `save_prev_target` of the DDP agent, and its pixels."""
    rows, columns = np.nonzero(seg[::-1] == 2)
    if not len(rows):
        return None
    return [float(-(rows.mean()-seg.shape[0]/2)*.1), float((columns.mean()-seg.shape[1]/2)*.1), int(len(rows))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--frontend', default=DEFAULT_WEIGHTS)
    parser.add_argument('--port', type=int, default=6100)
    parser.add_argument('--dump', type=Path, help='write segmentation and images of the first plans here')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    planner = Planner(args.checkpoint, args.frontend)
    print('planner ready', flush=True)
    with Listener(('127.0.0.1', args.port), authkey=AUTHKEY) as listener:
        while True:
            try:
                connection = listener.accept()
            except (EOFError, OSError, AuthenticationError):    # a client gone before the handshake ended
                continue
            with connection:
                count = 0
                while True:
                    try:
                        state = json.loads(connection.recv_bytes().decode())
                        if state.get('close'):
                            return
                        images = np.frombuffer(connection.recv_bytes(), np.uint8).reshape(SHAPE)
                    except (EOFError, ConnectionError):
                        break
                    reply, seg = planner.plan(images, state)
                    connection.send_bytes(json.dumps(reply).encode())
                    if args.dump and count < 40:
                        args.dump.mkdir(parents=True, exist_ok=True)
                        Image.fromarray((seg*120).astype(np.uint8)).save(args.dump/f'seg_{count:03d}.png')
                        Image.fromarray(np.concatenate(list(images[:3]), 1)).save(args.dump/f'front_{count:03d}.png')
                        Image.fromarray(np.concatenate(list(images[3:]), 1)).save(args.dump/f'back_{count:03d}.png')
                        (args.dump/f'state_{count:03d}.json').write_text(json.dumps(dict(state=state, reply=reply)))
                    count += 1


if __name__ == '__main__':
    main()
