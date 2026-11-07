"""Stream CarlaParking archives and write the Fused BEV cache used by sync_flow.

Workers read archives from Hugging Face in parallel while the main thread runs the
frozen front end. Only cameras, measurements, goals and `topdown` reach the disk;
voxels, depth and video are skipped. A task gets fused_bev/%04d.npy (fp16,
[267,16,16]), numbered by position in sorted measurements to match
sync_flow.windows.load_task. With --drop-images the cameras of a task are removed
once its cache is complete. Finished archives are recorded, so a rerun continues.
Topdown IoU is sampled per task to catch data that does not fit the model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddp_frontend import rig
from ddp_frontend import DEFAULT_WEIGHTS
from ddp_frontend.model import DdpFrontend

KEEP = re.compile(r'^(?:\./)?([\w-]+)/(task\d+)/((?:camera_(?:front|back)(?:_left|_right)?|measurements|parking_goal|topdown))/([\w.]+)$')
BATCH, IOU_EVERY, MIN_FREE_GB = 8, 20, 15


def token():
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN'].strip()
    path = Path.home()/'.cache'/'huggingface'/'token'
    if not path.exists():
        sys.exit('No Hugging Face login found. Run the login step first.')
    return path.read_text().strip()


def request(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'Authorization': f'Bearer {token()}', 'User-Agent': 'ddp-frontend-cache'}), timeout=120)


def archives(repo, revision, folder):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{f'{revision}/{folder}'.rstrip('/')}?recursive=false&limit=1000"
    with request(url) as response:
        items = [(item['path'], item['size']) for item in json.load(response) if item['path'].endswith('.tar.gz')]
    return sorted(items, key=lambda item: item[1])


class Counted:
    def __init__(self, stream):
        self.stream, self.count = stream, 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.count += len(data)
        return data


def extract(repo, revision, path, size, root, log):
    """Write the wanted members under root/<run>/<task>. Names are matched, never trusted."""
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}'
    started, written, runs = time.time(), 0, set()
    with request(url) as response:
        stream = Counted(response)
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                match = KEEP.match(member.name)
                if not member.isfile() or not match:
                    continue
                run, task, folder, name = match.groups()
                target = root/run/task/folder/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                written += member.size
                runs.add(run)
        while stream.read(1 << 20):
            pass
    if stream.count != size:
        raise IOError(f'{path}: read {stream.count} of {size} bytes')
    if not runs:
        raise IOError(f'{path}: no task files matched')
    log(f'{path}: {size/1e9:.1f} GB read, {written/1e9:.2f} GB kept, {time.time()-started:.0f} s')
    return sorted(runs)


def load_frame(task, name):
    return torch.stack([rig.prepare_image(Image.open(task/camera/f'{name}.png')) for camera in rig.CAMERAS])


def cache_task(model, task, log, drop_images=False):
    goal = json.loads((task/'parking_goal'/'0001.json').read_text())
    paths = sorted((task/'measurements').glob('*.json'))
    output = task/'fused_bev'
    output.mkdir(exist_ok=True)
    intrinsics, extrinsics = rig.intrinsics().cuda(), rig.extrinsics().cuda()
    vehicle, slot = [], []
    with ThreadPoolExecutor(8) as pool, torch.no_grad():
        for start in range(0, len(paths), BATCH):
            chunk = paths[start:start+BATCH]
            images = torch.stack(list(pool.map(lambda path: load_frame(task, path.stem), chunk))).cuda()
            targets = torch.stack([rig.target_in_ego(json.loads(path.read_text()), goal) for path in chunk])
            count = len(chunk)
            out = model(images, intrinsics.expand(count, -1, -1, -1), extrinsics.expand(count, -1, -1, -1),
                        targets.cuda())
            fused = out['fused_bev'].half()
            if not torch.isfinite(fused).all():
                raise ValueError(f'non-finite BEV in {task}')
            fused = fused.cpu().numpy()
            for offset, path in enumerate(chunk):
                index = start+offset
                np.save(output/f'{index:04d}.npy', fused[offset])
                if index % IOU_EVERY == 0:
                    labels = rig.segmentation_labels(task/'topdown'/f'encoded_{path.stem}.png', targets[offset].numpy())
                    prediction = out['segmentation'][offset].argmax(0).cpu().numpy()
                    vehicle.append(rig.iou(prediction, labels, 1))
                    slot.append(rig.iou(prediction, labels, 2))
    written = len(list(output.glob('[0-9]*.npy')))
    if written != len(paths) or not paths:
        raise ValueError(f'{task}: {written} cached of {len(paths)} frames')
    summary = dict(task=str(task), frames=len(paths), vehicle_iou=float(np.nanmean(vehicle)),
                   slot_iou=float(np.nanmean(slot)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    if drop_images:
        for camera in rig.CAMERAS:
            shutil.rmtree(task/camera)
    log(f"{task.parent.name}/{task.name}: {len(paths)} frames, vehicle {summary['vehicle_iou']:.3f}, "
        f"slot {summary['slot_iou']:.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--repo', default='ai4ce-drive/CarlaParking')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--folder', default='parkinglot_1000', help='repo folder, empty for the repo root')
    parser.add_argument('--name', help='output folder under --root, default --folder')
    parser.add_argument('--limit', type=int, help='process only the N smallest archives')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--drop-images', action='store_true')
    parser.add_argument('--local', nargs='+', type=Path, help='cache existing task folders, no download')
    args = parser.parse_args()
    model = DdpFrontend.load(args.weights)
    if args.local:
        for task in args.local:
            cache_task(model, task, print, args.drop_images)
        return
    root = args.root/(args.name or args.folder)
    root.mkdir(parents=True, exist_ok=True)
    logfile, lock = root/'build.log', threading.Lock()

    def log(text):
        line = f'{time.strftime("%H:%M:%S")} {text}'
        with lock:
            print(line, flush=True)
            with logfile.open('a', encoding='utf-8') as handle:
                handle.write(line+'\n')

    every = archives(args.repo, args.revision, args.folder)[:args.limit]
    marker = lambda path: root/(Path(path).name.removesuffix('.tar.gz')+'.done.json')
    todo = [item for item in every if not marker(item[0]).exists()]
    log(f'{len(every)} archives, {len(todo)} to do, {sum(size for _, size in todo)/1e9:.1f} GB, {args.workers} workers')
    ready, pending, failed = queue.Queue(maxsize=args.workers), queue.Queue(), []
    for item in todo:
        pending.put(item)

    def worker():
        while True:
            try:
                path, size = pending.get_nowait()
            except queue.Empty:
                return
            while shutil.disk_usage(root).free/1e9 < MIN_FREE_GB:
                log(f'waiting for disk space before {path}')
                time.sleep(60)
            stem = Path(path).name.removesuffix('.tar.gz')
            for attempt in range(3):
                try:
                    shutil.rmtree(root/stem, ignore_errors=True)
                    ready.put((path, extract(args.repo, args.revision, path, size, root, log)))
                    break
                except (OSError, tarfile.TarError, EOFError) as error:
                    log(f'{path}: attempt {attempt+1} failed: {error}')
                    time.sleep(30)
            else:
                shutil.rmtree(root/stem, ignore_errors=True)
                failed.append(path)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for thread in threads:
        thread.start()
    started, frames = time.time(), 0
    while any(thread.is_alive() for thread in threads) or not ready.empty():
        try:
            path, runs = ready.get(timeout=5)
        except queue.Empty:
            continue
        summaries, broken = [], []
        for run in runs:
            for task in sorted((root/run).iterdir()):
                try:
                    summaries.append(cache_task(model, task, log, args.drop_images))
                except (OSError, ValueError, KeyError) as error:
                    broken.append(str(task))
                    log(f'{task}: skipped: {error}')
        marker(path).write_text(json.dumps(dict(tasks=summaries, broken=broken), indent=2))
        frames += sum(item['frames'] for item in summaries)
        log(f'{path}: done, {frames} frames this run, {(time.time()-started)/3600:.2f} h')
    log(f'finished, failed archives: {failed}')


if __name__ == '__main__':
    main()
