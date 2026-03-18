# Double Pendulum: HNN vs Standard Neural Network

A small research-style codebase for comparing a **Hamiltonian Neural Network (HNN)** against a **standard neural network (NN)** on the **double pendulum** in canonically-conjugate coordinates.

---

## What this repo does

This project learns the dynamics of a conservative nonlinear system from data and compares two modeling approaches:

1. **Baseline NN**: directly predicts the state derivatives $\dot z$ from the current state.  
2. **HNN**: learns a scalar Hamiltonian $H_{\phi}(z)$ and induces dynamics through Hamilton's equations.

### Comparison metrics

- Derivative prediction error (vector-field accuracy)
- Rollout trajectory error (long-term accuracy)
- Energy drift over time (conservation behavior)
- Qualitative trajectory stability and phase portraits

---

## Mathematical setup

We use the canonical state

$$ z = (\theta_1, \theta_2, p_1, p_2) $$
$$ \dot{z} = J \nabla H(z) $$
$$ J = \left[\begin{array}{cccc}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
-1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0
\end{array}\right] $$

---

## Requirements

- Python 3.11+ (3.10 may work, but 3.11 is recommended)
- PyTorch 2.1+ (CPU or GPU)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart (run the full pipeline)

### 1) Generate dataset

```bash
python scripts/generate_dataset.py
```

### 2) Train baseline NN

```bash
python scripts/train_baseline.py
```

### 3) Train HNN

```bash
python scripts/train_hnn.py
```

### 4) Evaluate rollouts and energy drift

```bash
python scripts/evaluate_models.py
```

> All scripts support `--help` to list command-line options (e.g., epochs, batch size, output folders, checkpoint paths, etc.).

---

## Project layout

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── generate_dataset.py
│   ├── train_baseline.py
│   ├── train_hnn.py
│   └── evaluate_models.py
├── src/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── double_pendulum.py
│   │   └── integrators.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_nn.py
│   │   └── hnn.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── losses.py
│   └── utils/
│       ├── __init__.py
│       ├── data.py
│       ├── plotting.py
│       └── seed.py
└
```

---

## Notes

- This repo uses **canonical coordinates** (angles + momenta), not angular velocities, because that is the mathematically correct setting for an HNN.
- A ground-truth Hamiltonian is implemented explicitly so we can generate trustworthy training targets.
- The HNN objective is to match the induced vector field $J \nabla H_\psi(z)$ to the true derivatives.

---
