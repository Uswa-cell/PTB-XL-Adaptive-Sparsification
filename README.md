# Adaptive Sparsification for ECG Classification (PTB-XL)

Multilabel ECG classification on PTB-XL using an SE-ResNet1D backbone with a
learned input-conditional routing gate, evaluated against compute-matched
static structured pruning baselines.

## The question

Adaptive sparsification routes each input record to a different pruned expert,
so "easy" ECGs use cheaper computation and hard ones use the full network. In
theory this gives a better accuracy/cost trade-off than pruning the network
statically to the same average compute budget.

This repository tests that claim under matched compute, matched seeds, and
decision criteria fixed before the confirmatory run.

## What the experiments show

On PTB-XL, adaptive routing does **not** beat compute-matched static pruning.

The router collapses: effectively only ~1.0–1.35 of 4 experts are used. Fewer
MACs did not translate into fewer milliseconds or millijoules on the target
hardware — which is the central negative result here, and the reason measured
energy rather than MAC-count proxies is treated as ground truth.

A companion study on the PAMAP2 activity-recognition dataset shows the opposite
outcome, where adaptive gating does beat its matched static baseline. The
asymmetry between per-record hard routing (PTB-XL) and per-seed training-adaptive
channel gating (PAMAP2) is the more interesting finding.

## Notebook structure

Everything lives in one self-contained notebook, `PTB-XL_Adaptive_Sparsification.ipynb`.
There is no separate `src/` package — the modules are inlined as cells so the
notebook runs top to bottom in a fresh kernel.

| Section | Contents |
|---|---|
| Setup & dataset | Dependency check, auto-detection and extraction of the PTB-XL zip, dataset validation |
| Configuration | All hyperparameters in one cell — no external config file |
| Utilities | Seeding, logging, metrics, FLOPs/energy accounting |
| Preprocessing | ECG filtering, augmentation, SCP-code to superclass multilabel mapping |
| Models | SE-ResNet1D backbone, structured pruning, difficulty estimator, adaptive gate |
| Training | Losses (weighted BCE / focal), balanced sampler, Trainer |
| Phase A | Data loading, cleaning, EDA |
| Phase B | Dense baseline |
| Phase C | Static structured pruning baselines (30/60/90%) |
| Phase D | Adaptive sparsification framework, routing distribution, anti-collapse diagnostics |
| Phase D.2 | Random and confidence-based routing baselines |
| Phase E | Ablation: learned difficulty estimator vs heuristic gate |
| Phase F | Three-way comparison, energy-by-difficulty, rare-class robustness |
| Phase F.1 | Loss-component ablation and energy-λ sweep |
| Phase G | Explainability (Grad-CAM, integrated gradients) |
| Phase H | Multi-seed robustness and statistical significance |
| Phase I | Real hardware latency benchmark (batch size 1) |
| Phase J | Difficulty vs executed computation |
| Phase K | Energy-objective ablation |
| Final validation | Five-seed confirmatory run against the pre-specified decision gates |

## Getting the data

PTB-XL is not included in this repository. Download it from PhysioNet:

    https://physionet.org/content/ptb-xl/

Place the downloaded zip (or the extracted folder) beside the notebook. The
dataset cell auto-detects and extracts it, so no paths need editing.

## Setup

Python 3.12 is required. On 3.14, wfdb, pandas, scikit-learn and seaborn fail
to build.

```bash
conda create -n ptbxl python=3.12
conda activate ptbxl
pip install torch numpy pandas scikit-learn scipy seaborn matplotlib wfdb pyyaml jupyter
jupyter notebook PTB-XL_Adaptive_Sparsification.ipynb
```

Run the cells top to bottom in a fresh kernel. Experiments were run on an
NVIDIA RTX 5080.

Phase H and the final five-seed validation retrain models from scratch several
times and are compute-heavy — reduce the epoch count in the configuration cell
for a quick smoke test.

## Limitations

- Energy and latency are measured on a single GPU. Numbers will not transfer
  directly to other GPUs or to edge accelerators.
- Decision gates were fixed before the confirmatory run and are reported as-is,
  including the four that fail.
- One seed's sparsity sweep can accept no level and produce no deployable
  model; this is reported as a reliability limitation rather than dropped.

## License

MIT
