from __future__ import annotations

import torch

from banr_saq.config import BANRSAQConfig

from .entropy_loss import logical_minimum_entropy_loss
from .logical_losses import logical_classification_loss, logical_prior_loss
from .rg_consistency import rg_consistency_loss
from .syndrome_consistency import syndrome_consistency_loss


def combined_loss(outputs: dict, batch: dict, cfg: BANRSAQConfig) -> tuple[torch.Tensor, dict]:
    lp = logical_prior_loss(outputs['prior_logits'], batch['y_class'])
    lc = logical_classification_loss(outputs['logical_logits'], batch['y_class'])
    ent = logical_minimum_entropy_loss(outputs['qubit_logits'], batch['error'], batch['L'])
    rg = rg_consistency_loss(outputs['scale_logits'], batch['y_class'])
    syn = syndrome_consistency_loss(outputs['qubit_logits'], batch['H_full'], batch['syndrome'])
    total = cfg.lambda_lp * lp + cfg.lambda_lc * lc + cfg.lambda_ent * ent + cfg.lambda_rg * rg + cfg.lambda_syn * syn
    return total, {'lp': lp.detach(), 'lc': lc.detach(), 'ent': ent.detach(), 'rg': rg.detach(), 'syn': syn.detach()}
