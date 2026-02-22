from __future__ import annotations

import argparse
import importlib



def check_required_dependencies() -> None:
    required = {
        'torch': 'pip install -r requirements.txt',
        'numpy': 'pip install -r requirements.txt',
        'scipy': 'pip install -r requirements.txt',
    }
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        hints = '\n'.join(f"  - {m}: {required[m]}" for m in missing)
        raise RuntimeError(
            'Missing required dependencies for evaluation:\n'
            + hints
            + '\nInstall all with: pip install -r requirements.txt'
        )


def evaluate_distance(d: int = 7, family: str = 'rotated', n_samples: int = 4, p: float = 0.1) -> dict:
    check_required_dependencies()
    import torch

    from banr_saq.config import BANRSAQConfig
    from banr_saq.data.batch_builder import make_single_sample
    from banr_saq.data.precompute import precompute_distance
    from banr_saq.models.banr_saq import BANRSAQ

    cfg = BANRSAQConfig()
    cache = precompute_distance(d, family=family)
    logical_classes = 2 ** cache['L'].shape[0]
    model = BANRSAQ(cfg, logical_classes=logical_classes)
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_samples):
            batch = make_single_sample(cache, p=p)
            out = model(batch)
            pred = int(torch.argmax(out['logical_logits']).item())
            correct += int(pred == int(batch['y_class'].item()))
    return {'distance': d, 'family': cache['family'], 'accuracy': correct / n_samples}


def main():
    parser = argparse.ArgumentParser(description='BANR-SAQ minimal evaluation sanity script.')
    parser.add_argument('--distance', type=int, default=7)
    parser.add_argument('--family', type=str, default='rotated', choices=['rotated', 'toric'])
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--p', type=float, default=0.1)
    args = parser.parse_args()
    print(evaluate_distance(d=args.distance, family=args.family, n_samples=args.samples, p=args.p))


if __name__ == '__main__':
    main()
