from __future__ import annotations

import argparse
import random

import torch

from banr_saq.config import BANRSAQConfig
from banr_saq.data.batch_builder import make_single_sample
from banr_saq.data.precompute import precompute_distance
from banr_saq.losses.combined_loss import combined_loss
from banr_saq.models.banr_saq import BANRSAQ


def build_p_grid(p_init: float, p_final: float, n_p: int) -> list[float]:
    if n_p <= 0:
        raise ValueError('--n-p must be >= 1')
    if p_init > p_final:
        raise ValueError('--p-init must be <= --p-final')
    if n_p == 1:
        return [float(p_init)]
    step = (p_final - p_init) / (n_p - 1)
    return [float(p_init + i * step) for i in range(n_p)]


def run_train(
    distances: list[int],
    family: str = 'rotated',
    p_init: float = 0.05,
    p_final: float = 0.2,
    n_p: int = 4,
    steps: int = 1,
    seed: int = 0,
):
    """Run a minimal multi-distance training loop.

    - Distance is sampled uniformly from `distances` each step.
    - Physical error probability is sampled uniformly from a binned grid
      spanning [p_init, p_final] with n_p bins.
    """
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

    losses = []
    sampled_distances = []
    sampled_ps = []
    for _ in range(steps):
        d = random.choice(distances)
        p = random.choice(p_grid)
        sampled_distances.append(d)
        sampled_ps.append(p)

        batch = make_single_sample(caches[d], p=p)
        out = model(batch)
        loss, _ = combined_loss(out, batch, cfg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

    return {
        'loss_mean': sum(losses) / max(1, len(losses)),
        'loss_last': losses[-1],
        'sampled_distances': sampled_distances,
        'sampled_ps': sampled_ps,
        'p_grid': p_grid,
        'family': family,
    }


def _parse_distances(distances_arg: str) -> list[int]:
    vals = [int(x.strip()) for x in distances_arg.split(',') if x.strip()]
    if not vals:
        raise ValueError('At least one distance is required in --distances.')
    return vals


def main():
    parser = argparse.ArgumentParser(description='BANR-SAQ minimal multi-distance training script.')
    parser.add_argument('--distances', type=str, default='5,7,9,11', help='Comma-separated distances, e.g. "5,7,9,11"')
    parser.add_argument('--family', type=str, default='rotated', choices=['rotated', 'toric'])
    parser.add_argument('--p-init', type=float, default=0.05, help='Initial physical error probability.')
    parser.add_argument('--p-final', type=float, default=0.2, help='Final physical error probability.')
    parser.add_argument('--n-p', type=int, default=4, help='Number of bins in [p-init, p-final].')
    parser.add_argument('--steps', type=int, default=1, help='Number of optimizer steps')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    distances = _parse_distances(args.distances)
    summary = run_train(
        distances=distances,
        family=args.family,
        p_init=args.p_init,
        p_final=args.p_final,
        n_p=args.n_p,
        steps=args.steps,
        seed=args.seed,
    )
    summary['data_generation'] = 'banr_saq/data/batch_builder.py:make_single_sample'
    print(summary)


if __name__ == '__main__':
    main()
