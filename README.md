# SAQ-Decoder: Stabilizer-Aware Quantum Error Correction Decoder


> **SAQ: Stabilizer-Aware Quantum Error Correction Decoder**  
> David Zenati, Eliya Nachmani  
> School of Electrical and Computer Engineering, Ben-Gurion University of the Negev  
> *Published at ICLR 2026*

---

## Overview

SAQ-Decoder is a unified framework for quantum error correction (QEC) decoding that simultaneously achieves **near Maximum Likelihood (ML) accuracy** and **linear computational scalability** in syndrome size. It combines a dual-stream transformer architecture with a novel differentiable logical loss and a constraint-enforcing post-processing stage (CPND).

**Key results:**
- Error thresholds of **18.6%** (toric, depolarizing) and **10.99%** (toric, independent), approaching ML bounds of 18.9% and 11.0%
- **5× fewer FLOPs** and **5× faster inference** than QECCT at L=10
- Near-constant **1.2–1.9M parameters** from L=4 to L=10 (vs. 6.7M for QECCT at L=10)
- State-of-the-art across toric codes, rotated surface codes, color codes, and repetition codes

---

## Method

SAQ-Decoder operates in three sequential stages:

### Stage 1 — Dual-Stream Representation Construction
The syndrome vector is embedded into two complementary token streams:
- **Syndrome Stream**: each stabilizer measurement is mapped to a learned embedding; a global token is prepended for long-range aggregation
- **Logical Stream**: a shallow MLP (`lp_head`) maps the syndrome to initial logical class logits, which seed the logical token stream

### Stage 2 — Syndrome-Logical Transformer Decoder (SLTD)
A shared-weight transformer with asymmetric attention processes both streams over N layers:
- **Syndrome self-attention** uses a topology-constrained mask `M_S` derived from the parity-check matrix, restricting each stabilizer to attend only its topological neighbours plus the global token
- **Logical cross-attention** is unrestricted — logical tokens attend globally over all updated syndrome tokens to resolve quantum degeneracy

### Stage 3 — Constraint-Projected Nullspace Descent (CPND)
Applied at inference only. Enforces exact syndrome consistency over GF(2) in two steps:
1. **Projection**: maps the raw prediction to a feasible recovery operator via a precomputed left inverse of the augmented matrix `[H; L]`
2. **Nullspace descent**: greedily traverses the constraint-preserving affine space using transformer log-likelihood ratios as reliability weights, minimising recovery operator weight in O(m) time

### Logical-Centric Loss
Training minimises a three-term objective:

```
L = λ_LP · L_LP + λ_LC · L_LC + λ_Ent · L_Ent
```

| Term | Description | Default weight |
|------|-------------|----------------|
| `L_LP` | Cross-entropy on shallow MLP logical prior | λ_LP = 0.2 |
| `L_LC` | Cross-entropy on transformer logical output | λ_LC = 1.0 |
| `L_Ent` | Differentiable GF(2) logical entropy loss | λ_Ent = 1.0 |

`L_Ent` approximates the discrete XOR constraint via Bernoulli parity distributions, enabling end-to-end gradient flow through the logical error rate objective.

---

## Installation

```bash
git clone https://github.com/DavidZenati/SAQ-Decoder.git
cd SAQ-Decoder
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- [galois](https://github.com/mhostetter/galois)

```bash
pip install torch numpy scipy galois
```

---

## Usage

### Training

```bash
python Main.py \
  --code_type toric \
  --code_L 6 \
  --noise_type depolarization \
  --epochs 400 \
  --batch_size 512 \
  --lr 3e-4 \
  --N_dec 6 \
  --d_model 128 \
  --h 16 \
  --lambda_loss_ent 1.0 \
  --lambda_loss_lc 1.0 \
  --lambda_loss_lp 0.2
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--code_L` | Lattice distance L | `4` |
| `--noise_type` | `independent` or `depolarization` | `independent` |
| `--epochs` | Number of training epochs | `200` |
| `--batch_size` | Training batch size | `128` |
| `--lr` | Initial learning rate | `5e-4` |
| `--N_dec` | Number of transformer layers | `6` |
| `--d_model` | Embedding dimension | `64` |
| `--h` | Number of attention heads | `16` |
| `--lambda_loss_ent` | Weight for entropy loss | `1.0` |
| `--lambda_loss_lc` | Weight for logical class loss | `1.0` |
| `--lambda_loss_lp` | Weight for logical prior loss | `0.2` |
| `--upper_phy_err` | Upper physical error rate boundary | `0.2` |
| `--lower_phy_err` | Lower physical error rate boundary | `0.01` |
| `--no_mask` | Disable attention masking (ablation) | `0` |

### Output Structure

Training saves models and logs to:
```
Final_Results_SAQ_Decoder/
└── {code_type}/
    └── Code_L_{L}/
        └── noise_model_{noise_type}/
            └── {timestamp}/
                ├── best_model
                ├── last_model
                └── logging.txt
```

### Example Training Configurations

**Toric code, L=8, depolarizing noise** (from paper):
```bash
python Main.py --code_type toric --code_L 8 --noise_type depolarization \
               --epochs 600 --batch_size 128 --lr 2e-4 --N_dec 8 --d_model 128
```

---

## Code Structure

```
SAQ-Decoder/
├── Main.py        # Training loop, dataset, argument parser
├── Model.py       # Stage 1 & 2: dual-stream transformer, loss functions
├── CPND.py        # Stage 3: constraint projection and nullspace descent
├── Codes.py       # Code definitions (H, L matrices)
└── README.md
```

| File | Contents |
|------|----------|
| `Model.py` | `SAQ_Transformer`, `Encoder`, `MultiHeadedAttention`, `PositionwiseFeedForward`, logical loss |
| `CPND.py` | `exact_left_inverse`, `kernel_basis`, `greedy_nullspace_refine` |
| `Codes.py` | code and noise utilities |
| `Main.py` | `QECC_Dataset`, `train()`, `test()`, CLI entry point |

---

## Results

### Error Thresholds (Depolarizing Noise)

| Method | Toric Code | Rotated Surface |
|--------|-----------|-----------------|
| **SAQ-Decoder (Ours)** | **18.6%** | **18.3%** |
| ML bound | 18.9% | — |
| QECCT | 17.8% | 17.2% |
| Astra | — | 17.0% |
| SU-NetQD | 16.3% | — |
| UIUF | 15.5% | 15.6% |
| MWPM | 16.0% | 14.0% |
| BPOSD-2 | 16.0% | 14.1% |

### Computational Efficiency (L=10, Depolarizing)

| Metric | SAQ-Decoder | QECCT | Speedup |
|--------|-------------|-------|---------|
| FLOPs | 0.80 G | 4.10 G | **5.1×** |
| Parameters | 1.85 M | 6.64 M | **3.6×** |
| Inference time | 0.45 ms | 2.33 ms | **5.2×** |

---

## Hardware

All experiments were conducted on a single **48GB NVIDIA L40S GPU**.  

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zenati2025saq,
  title={SAQ: Stabilizer-Aware Quantum Error Correction Decoder},
  author={Zenati, David and Nachmani, Eliya},
  journal={arXiv preprint arXiv:2512.08914},
  year={2025}
}
```

---

## Acknowledgements

We build on the QECCT decoder implementation from [Choukroun & Wolf (2023)](https://github.com/yoniLc/DQEC/), we also build on the toric code implementation from [Krastanov & Jiang (2017)](https://github.com/Krastanov/neural-decoder/) 
