"""Write weights/ddp_frontend.pt from the public DDP checkpoint.

Usage: export_ddp_frontend.py LAST_CKPT [--out weights/ddp_frontend.pt]

LAST_CKPT is last.ckpt of huggingface.co/ChampagneAndfragrance/Dino_Diffusion_Parking.
Only the perception modules are kept; the Diffusion planner is left out.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
KEEP = ('bev_model', 'bev_encoder', 'feature_fusion', 'segmentation_head', 'film_modulate')


def read(path):
    """The checkpoint pickles tool.config.Configuration; a stub keeps weights_only on."""
    class Configuration:
        pass
    Configuration.__module__, Configuration.__qualname__ = 'tool.config', 'Configuration'
    sys.modules['tool'] = types.ModuleType('tool')
    sys.modules['tool.config'] = types.ModuleType('tool.config')
    sys.modules['tool.config'].Configuration = Configuration
    with torch.serialization.safe_globals([Configuration]):
        return torch.load(path, map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'weights'/'ddp_frontend.pt')
    args = parser.parse_args()
    saved = read(args.checkpoint)
    state = {}
    for key, value in saved['state_dict'].items():
        name = key.split('.', 1)[1] if key.startswith('parking_model.') else key
        if name.split('.')[0] in KEEP:
            state[name] = value
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    hyper = saved['hyper_parameters']
    config = hyper.get('cfg', hyper.get('config', hyper))
    config = dict(vars(config)) if not isinstance(config, dict) else config
    torch.save(dict(state_dict=state, config=config,
                    source=dict(repo='ChampagneAndfragrance/Dino_Diffusion_Parking', file='last.ckpt', sha256=digest,
                                epoch=saved.get('epoch'), global_step=saved.get('global_step'))), args.out)
    print(f'{len(state)} tensors -> {args.out}')


if __name__ == '__main__':
    main()
