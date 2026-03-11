<div align="center">

# Chem-Gym: SAGCM — Reinforcement Learning for Operando Catalyst Surface Optimization

**Surface-Adsorbate Grand Canonical Modeling (SAGCM)**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.7-orange.svg)](https://stable-baselines3.readthedocs.io/)

</div>

---

## Overview

**Chem-Gym** implements the **SAGCM (Surface-Adsorbate Grand Canonical Modeling)** framework — an end-to-end Reinforcement Learning pipeline for optimizing bimetallic catalyst surfaces under operando conditions. The agent learns to mutate atomic sites on a Cu-Pd alloy slab to minimize the **Grand Potential (Ω)**, directly encoding thermodynamic stability under realistic CO gas environments.

The framework integrates:

- **Maskable PPO** with a physics-informed residual policy (**PIRP**) for constrained combinatorial optimization
- **Dual ML-potential oracles** (UMA for slab thermodynamics + EquiformerV2/OC25 ensemble for adsorbate energetics) via FAIRChem
- **Grand Canonical reward shaping** with PID-Lagrangian composition constraints, UMA-based PBRS, and debt-guided exploration

### Key Scientific Features

| Feature                                | Description                                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Grand Potential Optimization** | $\Omega = E_{\text{slab}} + E_{\text{ads}} - N_{\text{CO}} \cdot \mu_{\text{CO}}$ as the RL objective |
| **Hybrid Oracle**                | UMA (slab energy) + OC25 ensemble (adsorbate energy) with uncertainty estimation                        |
| **PIRP**                         | Physics-Informed Residual Policy — injects analytical Pd-segregation drives into actor logits          |
| **Greedy CO Placer**             | Langmuir-isotherm-guided CO placement with lateral repulsion                                            |
| **PID-Lagrangian Constraint**    | Adaptive soft penalty keeping Pd fraction near the bulk target                                          |
| **UMA-PBRS**                     | Potential-Based Reward Shaping using UMA energy as the potential function                               |
| **Deviation Mask**               | Hard action mask enforcing composition envelopes (Route A ablation)                                     |
| **Crystal GNN Extractor**        | Message-passing graph neural network for structural observation encoding                                |

---

## Repository Structure

```text
ChemGymProject/
├── ProjectMain/
│   ├── main.py                          # CLI entry point (train / eval / baseline)
│   ├── requirements.txt                 # Pinned dependencies
│   ├── Download.py                      # UMA reference file downloader
│   ├── checkpoints/                     # Model weights & reference data
│   │   ├── uma-s-1p1.pt                #   UMA-S v1.1 checkpoint
│   │   └── references/                 #   Element reference energies (YAML)
│   │
│   ├── chem_gym/                        # Core library
│   │   ├── config.py                    #   Centralized dataclass configs
│   │   ├── baselines.py                 #   Random search & simulated annealing
│   │   ├── envs/
│   │   │   └── chem_env.py              #   Gymnasium RL environment (ChemGymEnv)
│   │   ├── agent/
│   │   │   ├── trainer.py               #   Training pipeline (SB3 MaskablePPO/PPO)
│   │   │   ├── pirp_policy.py           #   Physics-Informed Residual Policy
│   │   │   └── graph_feature_extractor.py  # Crystal GNN feature extractor
│   │   ├── surrogate/
│   │   │   ├── ocp_model.py             #   FAIRChem oracle wrappers (UMA, EqV2, OC25)
│   │   │   ├── hybrid_oracle.py         #   Hybrid Grand Potential oracle
│   │   │   └── ensemble.py              #   Lightweight surrogate ensemble
│   │   ├── physics/
│   │   │   ├── analytical_prior.py      #   Pd-segregation analytical driving force
│   │   │   ├── co_placer.py             #   Greedy CO placement with Langmuir model
│   │   │   ├── pid_lagrangian.py        #   PID-Lagrangian composition controller
│   │   │   └── uma_shaping.py           #   UMA potential-based reward shaping
│   │   └── analysis/
│   │       ├── vis_callback.py          #   Training visualization callback
│   │       ├── advanced_vis.py          #   Plotly 3D interactive structure viewer
│   │       ├── sro_analysis.py          #   Warren-Cowley SRO parameter calculation
│   │       └── pmg_utils.py             #   Pymatgen utility helpers
│   │
│   ├── week2_ev_multirun.py             # Multi-seed explained variance analysis
│   ├── week2_smoke_compare.py           # PIRP vs Vanilla comparison smoke test
│   ├── week3_phase_scan_minimal.py      # μ_CO phase diagram scanning experiment
│   ├── week3_delta_diagnostic.py        # ΔΩ energy consistency diagnostics
│   ├── week3_oracle_calibration.py      # Oracle vs. DFT literature benchmarks
│   ├── week3_reward_v2_eval.py          # Reward v2 Spearman correlation analysis
│   └── paper_vis_training.py            # Publication-quality training curve plots
│
├── checkpoints/                         # Archived experiment results
│   ├── week2_ev_multirun/               #   Multi-seed EV curves
│   └── week3_phase_scan_minimal/        #   μ_CO phase scan CSV & trained models
├── chem_gym_tensorboard/                # TensorBoard event logs
├── LICENSE                              # Apache 2.0 + SAGCM IP declaration
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 (recommended via Conda)
- CUDA 12.8 (for GPU acceleration; CPU fallback supported)

### 1. Create Environment

```bash
conda create -n chemgym python=3.11 -y
conda activate chemgym
```

### 2. Install PyTorch

**GPU (CUDA 12.8):**

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**CPU only:**

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r ProjectMain/requirements.txt
```

### 4. Download Model Weights

The UMA checkpoint (`uma-s-1p1.pt`) and element reference files must be placed under `ProjectMain/checkpoints/`. Reference files can be fetched via:

```bash
python ProjectMain/Download.py
```

### 5. Verify Installation

```bash
python -m pip check
python ProjectMain/main.py --help
```

---

## Quick Start

> **Important:** All commands should be executed from the repository root directory (`ChemGymProject/`).

### Smoke Test (Recommended First Run)

A minimal training run to verify the setup end-to-end:

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode image \
  --total-steps 2048 \
  --n-active-layers 2 \
  --learning-rate 1e-4 \
  --device cpu \
  --use-masking \
  --save-dir ProjectMain/checkpoints/smoke_test
```

> **Note:** The PPO rollout buffer size (`n_steps`) is configurable via `--ppo-n-steps` (default 512). The actual training will proceed in rollout increments of this size.

---

## Usage

### Training

#### MaskablePPO with PIRP (Recommended)

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode graph \
  --total-steps 40000 \
  --n-active-layers 3 \
  --learning-rate 1e-4 \
  --device cuda \
  --use-masking \
  --use-pirp
```

#### Standard MaskablePPO

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode graph \
  --total-steps 40000 \
  --n-active-layers 3 \
  --learning-rate 1e-4 \
  --device cuda \
  --use-masking
```

#### Standard PPO (No Action Masking)

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode graph \
  --total-steps 40000 \
  --n-active-layers 3 \
  --learning-rate 1e-4 \
  --device cuda
```

#### Route A Experiment Profile

A curated configuration with hard composition masks + debt shaping + no UMA-PBRS:

```bash
python ProjectMain/main.py \
  --mode train \
  --experiment-profile route_a \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode graph \
  --total-steps 40000 \
  --device cuda \
  --use-masking
```

### Baseline Comparisons

```bash
python ProjectMain/main.py \
  --mode baseline \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --total-steps 40000
```

This runs random search and simulated annealing for comparison.

### Evaluation

```bash
python ProjectMain/main.py \
  --mode eval \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --load-dir ProjectMain/checkpoints/<your_run_dir> \
  --use-masking
```

**Evaluation outputs:**

- `best_optimized.xyz` — Optimized atomic structure
- Visualization files (`.html` / `.png` / `.cif`) depending on callback configuration

---

## Key CLI Arguments

| Argument                 | Default     | Description                                                |
| ------------------------ | ----------- | ---------------------------------------------------------- |
| `--mode`               | `train`   | `train` / `eval` / `baseline`                        |
| `--obs-mode`           | `graph`   | Observation mode:`graph` (GNN) or `image` (grid)       |
| `--total-steps`        | `5000`    | Total environment steps                                    |
| `--use-masking`        | `False`   | Enable action masking (MaskablePPO)                        |
| `--use-pirp`           | `False`   | Enable Physics-Informed Residual Policy                    |
| `--n-active-layers`    | `4`       | Number of mutable surface layers                           |
| `--mu-co`              | `-1.0`    | CO chemical potential (eV)                                 |
| `--bulk-pd-fraction`   | `0.08`    | Target bulk Pd composition fraction                        |
| `--oracle-mode`        | `auto`    | Oracle selection:`auto` / `uma` / `eq2` / `hybrid` |
| `--experiment-profile` | `default` | Parameter bundle:`default` / `route_a`                 |
| `--device`             | `auto`    | Compute device:`cuda` / `cpu` / `auto`               |
| `--seed`               | `None`    | Random seed for reproducibility                            |
| `--enable-vis`         | `False`   | Enable periodic structure visualization during training    |

<details>
<summary><b>Advanced Reward & Constraint Arguments</b></summary>

| Argument                        | Default     | Description                                       |
| ------------------------------- | ----------- | ------------------------------------------------- |
| `--omega-reward-scale`        | `1.0`     | Scale factor for ΔΩ reward term                 |
| `--delta-omega-scale`         | `1.0`     | Scale for delta-omega component                   |
| `--debt-improvement-scale`    | `0.1`     | Composition debt improvement reward               |
| `--debt-abs-penalty`          | `0.0`     | Absolute debt penalty                             |
| `--step-penalty`              | `0.01`    | Per-step penalty                                  |
| `--linear-reward-clip`        | `3.0`     | Reward clipping bound                             |
| `--constraint-threshold-frac` | `0.12`    | Composition violation threshold                   |
| `--constraint-weight`         | `1.0`     | PID-Lagrangian penalty weight                     |
| `--constraint-update-mode`    | `rollout` | Lambda update:`step` / `rollout` / `frozen` |
| `--disable-uma-pbrs`          | `False`   | Disable UMA potential-based reward shaping        |
| `--uma-pbrs-scale`            | `50.0`    | PBRS potential scaling                            |
| `--pirp-scale`                | `0.02`    | PIRP logit injection strength                     |
| `--pirp-anneal-schedule`      | `cosine`  | PIRP annealing:`cosine` / `linear`            |
| `--enable-deviation-mask`     | `False`   | Hard composition envelope mask                    |
| `--disable-co-adsorption`     | `False`   | Disable CO adsorption modeling                    |

</details>

<details>
<summary><b>Oracle & Surrogate Arguments</b></summary>

| Argument                  | Default                             | Description                                  |
| ------------------------- | ----------------------------------- | -------------------------------------------- |
| `--oracle-ckpt`         | `None`                            | Backward-compatible alias for `--eq2-ckpt` |
| `--uma-ckpt`            | `checkpoints/uma-s-1p1.pt`        | UMA model checkpoint path                    |
| `--eq2-ckpt`            | `checkpoints/eq2_83M_2M.pt`       | EquiformerV2 checkpoint path                 |
| `--ads-sm-ckpt`         | `checkpoints/esen_sm_conserve.pt` | OC25 small model checkpoint                  |
| `--ads-md-ckpt`         | `checkpoints/esen_md_direct.pt`   | OC25 medium model checkpoint                 |
| `--ads-task`            | `oc25`                            | Adsorbate task type:`oc25` / `oc20`      |
| `--oracle-fmax`         | `0.05`                            | LBFGS force convergence threshold            |
| `--oracle-max-steps`    | `100`                             | Maximum relaxation steps                     |
| `--uncertainty-penalty` | `0.0`                             | Surrogate uncertainty penalty weight         |
| `--oracle-threshold`    | `None`                            | Uncertainty threshold for oracle fallback    |

</details>

---

## Experiment Scripts

Beyond the main training pipeline, the repository includes dedicated experiment scripts:

| Script                          | Purpose                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| `week2_ev_multirun.py`        | Multi-seed training with explained variance tracking                 |
| `week2_smoke_compare.py`      | PIRP vs. Vanilla MaskablePPO A/B comparison                          |
| `week3_phase_scan_minimal.py` | μ_CO phase diagram scan across multiple chemical potentials         |
| `week3_delta_diagnostic.py`   | ΔΩ energy consistency diagnostics between UMA and OC backends      |
| `week3_oracle_calibration.py` | Oracle prediction vs. DFT literature benchmark (Cu/Pd CO adsorption) |
| `week3_reward_v2_eval.py`     | Reward v2 evaluation with Spearman correlation analysis              |
| `paper_vis_training.py`       | Publication-ready training curve visualization from TensorBoard logs |

---

## Architecture

### RL Environment (`ChemGymEnv`)

The environment models a Cu-Pd(111) alloy slab under CO gas. At each step, the agent selects a `(site, element)` mutation or a no-op action. The environment:

1. Applies the atomic mutation to the slab
2. Runs **Greedy CO Placement** (Langmuir-guided with lateral repulsion)
3. Evaluates energy via the designated oracle backend
4. Computes the Grand Potential: $\Omega = E_{\text{slab}} + E_{\text{ads}} - N_{\text{CO}} \cdot \mu_{\text{CO}}$
5. Assembles the reward from: ΔΩ (main term) + debt shaping + PID-Lagrangian penalty + UMA-PBRS + step penalty

### Oracle Backends

```
                    ┌─────────────────────────┐
                    │     HybridOracle         │
                    │  (Grand Potential Ω)     │
                    └──────┬──────────┬────────┘
                           │          │
              ┌────────────▼──┐  ┌────▼────────────┐
              │   UMA Oracle  │  │ OC25 Ensemble    │
              │ (slab energy) │  │ (ads. energy)    │
              └───────────────┘  └─────────────────┘
                                         │
                              ┌───────────▼──────────┐
                              │  EquiformerV2 Oracle  │
                              │  (legacy fallback)    │
                              └───────────────────────┘
```

- **UMA Oracle**: Uses FAIRChem's Universal Materials Accelerator for slab-level thermodynamics and formation energy
- **OC25 Ensemble Oracle**: Multi-checkpoint ensemble for adsorbate system energy with uncertainty quantification
- **Hybrid Oracle**: Combines both to avoid cross-backend reference energy mismatch
- **EMT Fallback**: ASE's Effective Medium Theory calculator when no ML checkpoint is available

### Policy Architecture

- **Crystal GNN Feature Extractor**: Multi-layer message-passing network with residual connections and layer normalization, operating on the atomic graph
- **PIRP (Physics-Informed Residual Policy)**: Adds analytical Pd-segregation driving forces as a learnable residual on actor logits:
  - $\text{logits} = \text{base\_logits} + \alpha \cdot v_i \cdot g_{i \to e}$
  - Where $g_{\text{Pd}}$ encodes surface energy, strain, and CO adsorption preference terms
  - A learnable site-gate network modulates injection strength per site

---

## Monitoring

### TensorBoard

Training logs are written to `./chem_gym_tensorboard/` by default:

```bash
tensorboard --logdir ./chem_gym_tensorboard/
```

Tracked metrics include:

- `rollout/ep_rew_mean` — Mean episode reward
- `train/explained_variance` — PPO value function quality
- Custom scalars: Ω, n_CO, Pd fraction, λ (constraint multiplier), noop ratio

### Output Artifacts

Each training run creates a timestamped directory:

```text
checkpoints/<run_name>/
├── model.zip              # SB3 model checkpoint
├── vec_normalize.pkl      # Observation normalization statistics
└── vis/                   # Visualization outputs (if --enable-vis)
    ├── step_XXXX.cif      # CIF crystal structures
    ├── step_XXXX.html     # Plotly 3D interactive views
    └── step_XXXX.png      # Top/side view static plots
```

---

## Platform Notes

### Windows (PowerShell)

```powershell
conda activate chemgym
$env:MPLBACKEND='Agg'   # Prevents Tk backend crashes
python ProjectMain\main.py --mode train --oracle-ckpt ProjectMain\checkpoints\uma-s-1p1.pt --obs-mode graph --total-steps 40000 --n-active-layers 3 --learning-rate 1e-4 --device cuda --use-masking
```

### Linux

```bash
conda activate chemgym
export MPLBACKEND=Agg
python ProjectMain/main.py --mode train --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt --obs-mode graph --total-steps 40000 --n-active-layers 3 --learning-rate 1e-4 --device cuda --use-masking
```

> **Tip:** Set `MPLBACKEND=Agg` to avoid matplotlib display issues in headless or remote environments.

---

## Dependencies

Core stack (see [requirements.txt](ProjectMain/requirements.txt) for pinned versions):

| Category                       | Packages                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| **RL**                   | `gymnasium`, `stable-baselines3[extra]`, `sb3-contrib` |
| **ML Potentials**        | `fairchem-core`, `torch-geometric`, `torch`            |
| **Materials Science**    | `ase`, `pymatgen`, `mp-api`                            |
| **Scientific Computing** | `numpy`, `scipy`, `pandas`, `matplotlib`             |
| **Visualization**        | `plotly`, `tensorboard`                                  |
| **Utilities**            | `tqdm`, `rich`, `pyyaml`, `requests`                 |

---

## License

This project is licensed under the **Apache License 2.0** with an additional intellectual property declaration for the SAGCM algorithm. See [LICENSE](LICENSE) for full details.

**Third-party weights:** The UMA model checkpoints and reference files are property of FAIR / Open Catalyst Project and are subject to their original license terms (CC-BY-NC 4.0 or MIT as applicable).

---

## Citation

If you use the SAGCM framework, Chem-Gym environment, or any of the specific algorithmic modules (PIRP, GreedyCOPlacer, UMAPotentialShaper, PID-Lagrangian constraint) in academic publications or derivative works, please cite this repository:

```bibtex
@software{xu2025sagcm,
  title   = {Chem-Gym: SAGCM -- Reinforcement Learning for Operando Catalyst Surface Optimization},
  author  = {Xu, Chengrui},
  year    = {2025},
  url     = {https://github.com/your-repo/ChemGymProject}
}
```
