from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class BANRSAQConfig:
    embed_dim: int = 128
    num_heads: int = 8
    ff_mult: int = 4
    boundary_embed_bins: int = 4
    dist_clip: int = 5
    use_qubit_stream: bool = True
    use_decoder: bool = True
    share_readout_probe: bool = True
    max_scales: int | None = None
    code_family: str = "rotated"

    # losses
    lambda_lp: float = 0.2
    lambda_lc: float = 1.0
    lambda_ent: float = 1.0
    lambda_rg: float = 0.3
    lambda_syn: float = 0.1

    # training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    min_lr: float = 1e-6
    train_distances: Sequence[int] = field(default_factory=lambda: (5, 7, 9, 11))
    p_init: float = 0.05
    p_final: float = 0.2
    n_p: int = 4
