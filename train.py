from __future__ import annotations

import argparse
import importlib
import random
from pathlib import Path


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
            'Missing required dependencies for training:\n'
            + hints
            + '\nInstall all with: pip install -r requirements.txt'
        )


def build_p_grid(p_init: float, p_final: float, n_p: int) -> list[float]:
    if n_p <= 0:
        raise ValueError('--n-p must be >= 1')
    if p_init > p_final:
        raise ValueError('--p-init must be <= --p-final')
    if n_p == 1:
        return [float(p_init)]
    step = (p_final - p_init) / (n_p - 1)
    return [float(p_init + i * step) for i in range(n_p)]


def _batch_loss(model, combined_loss, cfg, make_single_sample, cache, p: float, batch_size: int):
    losses = []
    for _ in range(batch_size):
        sample = make_single_sample(cache, p=p)
        out = model(sample)
        loss, _ = combined_loss(out, sample, cfg)
        losses.append(loss)
    return sum(losses) / len(losses)


def run_train(
    distances: list[int],
    family: str = 'rotated',
    p_init: float = 0.05,
    p_final: float = 0.2,
    n_p: int = 4,
    epochs: int = 1,
    n_batches_per_p: int = 1,
    batch_size: int = 8,
    seed: int = 0,
    log_file: str = 'train_log.txt',
):
    """Epoch training schedule that covers full p-range for each distance.

    For each epoch, iterate all distances, all p bins, and N batches per p-bin.
    Total optimizer steps per epoch = len(distances) * n_p * n_batches_per_p.
    """
    check_required_dependencies()
    import torch

    from banr_saq.config import BANRSAQConfig
    from banr_saq.data.batch_builder import make_single_sample
    from banr_saq.data.precompute import precompute_distance
    from banr_saq.losses.combined_loss import combined_loss
    from banr_saq.models.banr_saq import BANRSAQ

    random.seed(seed)
    p_grid = build_p_grid(p_init, p_final, n_p)

    cfg = BANRSAQConfig(
        code_family=family,
        train_distances=tuple(distances),
        p_init=p_init,
        p_final=p_final,
        n_p=n_p,
    )

    caches = {d: precompute_distance(d, family=family) for d in distances}
    logical_classes = 2 ** next(iter(caches.values()))['L'].shape[0]

    model = BANRSAQ(cfg, logical_classes=logical_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path = Path(log_file)
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as f:
        f.write('epoch,steps,loss_mean,loss_min,loss_max,distances,p_grid,batch_size,n_batches_per_p\n')

    epoch_stats = []
    for epoch in range(1, epochs + 1):
        step_losses = []
        work_items = [(d, p, b) for d in distances for p in p_grid for b in range(n_batches_per_p)]
        random.shuffle(work_items)

        for d, p, _ in work_items:
            loss = _batch_loss(model, combined_loss, cfg, make_single_sample, caches[d], p=p, batch_size=batch_size)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            step_losses.append(float(loss.item()))

        stats = {
            'epoch': epoch,
            'steps': len(work_items),
            'loss_mean': sum(step_losses) / max(1, len(step_losses)),
            'loss_min': min(step_losses) if step_losses else 0.0,
            'loss_max': max(step_losses) if step_losses else 0.0,
            'distances': distances,
            'p_grid': p_grid,
            'batch_size': batch_size,
            'n_batches_per_p': n_batches_per_p,
        }
        epoch_stats.append(stats)

        with log_path.open('a', encoding='utf-8') as f:
            f.write(
                f"{stats['epoch']},{stats['steps']},{stats['loss_mean']:.6f},{stats['loss_min']:.6f},{stats['loss_max']:.6f},"
                f"\"{stats['distances']}\",\"{stats['p_grid']}\",{batch_size},{n_batches_per_p}\n"
            )

    return {
        'family': family,
        'epochs': epochs,
        'epoch_stats': epoch_stats,
        'p_grid': p_grid,
        'log_file': str(log_path),
    }


def _parse_distances(distances_arg: str) -> list[int]:
    vals = [int(x.strip()) for x in distances_arg.split(',') if x.strip()]
    if not vals:
        raise ValueError('At least one distance is required in --distances.')
    return vals


def main():
    parser = argparse.ArgumentParser(description='BANR-SAQ training script with full p-range coverage per epoch.')
    parser.add_argument('--distances', type=str, default='5,7,9,11', help='Comma-separated distances, e.g. "5,7,9,11"')
    parser.add_argument('--family', type=str, default='rotated', choices=['rotated', 'toric'])
    parser.add_argument('--p-init', type=float, default=0.05, help='Initial physical error probability.')
    parser.add_argument('--p-final', type=float, default=0.2, help='Final physical error probability.')
    parser.add_argument('--n-p', type=int, default=4, help='Number of bins in [p-init, p-final].')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--n-batches-per-p', type=int, default=1, help='Number of batches per p-bin for each distance each epoch.')
    parser.add_argument('--batch-size', type=int, default=8, help='Samples per batch.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-file', type=str, default='train_log.txt')
    args = parser.parse_args()

    distances = _parse_distances(args.distances)
    summary = run_train(
        distances=distances,
        family=args.family,
        p_init=args.p_init,
        p_final=args.p_final,
        n_p=args.n_p,
        epochs=args.epochs,
        n_batches_per_p=args.n_batches_per_p,
        batch_size=args.batch_size,
        seed=args.seed,
        log_file=args.log_file,
    )
    summary['data_generation'] = 'banr_saq/data/batch_builder.py:make_single_sample'
    print(summary)


if __name__ == '__main__':
    main()
