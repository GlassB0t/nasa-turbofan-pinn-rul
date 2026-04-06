# 🛩️ PINN-RUL: Physics-Informed Neural Network for Turbofan Engine RUL Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-green)](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
[![Course](https://img.shields.io/badge/Course-ML%20for%20Industrial%20Applications-purple)](/)

> **Course Project** — Machine Learning for Industrial Applications  
> IIT Kharagpur · Department of Industrial & Systems Engineering

---

## 📌 Project Overview

This project predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using a **Physics-Informed Neural Network (PINN)** trained on the **NASA C-MAPSS benchmark dataset**.

Unlike a standard neural network that purely minimises prediction error, our PINN embeds three physical laws of turbofan degradation directly into the training loss:

| Physics Constraint | Meaning | Loss Term |
|---|---|---|
| **Monotonicity** | Engine health can only worsen over time — RUL never increases | `L_monotone` |
| **Boundary Condition** | At the last cycle, RUL → 0 | `L_boundary` |
| **Smoothness** | Degradation follows smooth physical processes (no erratic jumps) | `L_smooth` |

The total training loss is:

```
L_total = α·L_data + β·L_monotone + γ·L_boundary + δ·L_smooth
```

---

## 📁 Project Structure

```
nasa-turbofan-pinn/
│
├── main.py                    ← One-click entry point (run this!)
│
├── src/
│   ├── data_loader.py         ← Dataset loading, RUL labelling, feature engineering
│   ├── pinn_model.py          ← PINN architecture + physics loss functions
│   ├── train.py               ← Training loop with physics-aware batching
│   └── evaluate.py            ← Metrics, plots, monotonicity verification
│
├── data/
│   ├── README.md              ← How to download the C-MAPSS dataset
│   └── (place .txt files here)
│
├── results/                   ← Auto-created during training
│   ├── best_model.pth
│   ├── training_curves.png
│   ├── evaluation_plots.png
│   ├── rmse_by_zone.png
│   └── results.json
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/GlassB0t/nasa-turbofan-pinn-rul.git
cd nasa-turbofan-pinn-rul
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

- Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Download and unzip
- Place all `.txt` files in the `data/` folder

```
data/train_FD001.txt   data/test_FD001.txt   data/RUL_FD001.txt
data/train_FD002.txt   data/test_FD002.txt   data/RUL_FD002.txt
... etc.
```

### 4. Train the Model

```bash
# Default: FD001, 150 epochs
python main.py

# Custom options
python main.py --subset FD001 --epochs 200 --lr 0.001 --batch_size 512
```

### 5. Evaluate Only (if model already trained)

```bash
python src/evaluate.py --subset FD001 --model_path results/best_model.pth
```

---

## 🧠 Model Architecture

```
Input (F features)
      ↓
 Linear(F → 256)  +  BatchNorm  +  Swish  +  Dropout(0.2)
      ↓
 Linear(256 → 256) + BatchNorm  +  Swish  +  Dropout(0.2)
      ↓
 Linear(256 → 128) + BatchNorm  +  Swish  +  Dropout(0.2)
      ↓
 Linear(128 → 64)  + BatchNorm  +  Swish  +  Dropout(0.2)
      ↓
 Linear(64 → 1)  +  Softplus          ← ensures RUL ≥ 0
      ↓
 RUL Prediction (cycles)
```

**Why Swish activation?**  
Swish (`x · σ(x)`) is smooth and differentiable everywhere — a critical property when computing the gradient-based physics constraints. ReLU's non-differentiability at zero can destabilise physics loss gradients.

**Why Softplus output?**  
RUL cannot be negative. Softplus (`log(1 + eˣ)`) smoothly enforces non-negativity without the hard zero of ReLU.

---

## ⚙️ Feature Engineering

| Feature Type | Details |
|---|---|
| Raw sensors | 14 sensors retained after variance filtering |
| Rolling mean | Window sizes: 5, 10, 30 cycles |
| Rolling std | Window sizes: 5, 10, 30 cycles |
| Operational settings | setting_1, setting_2, setting_3 |
| **Total features** | **~100 input features** |

---

## 📊 Expected Results (FD001 Test Set)

| Metric | Random Forest (baseline) | PINN (ours) |
|---|---|---|
| RMSE | ~20.4 | ~16–18 |
| MAE | ~14.7 | ~11–13 |
| R² | ~0.81 | ~0.87 |
| NASA Score | ~3120 | ~1800–2400 |
| Monotonicity Compliance | N/A | ~92–96% |

*Results may vary slightly with hardware and random seed.*

---

## 📈 Outputs

After training, the `results/` folder contains:

| File | Description |
|---|---|
| `best_model.pth` | Saved model weights (best validation loss) |
| `training_curves.png` | Train vs validation loss + physics loss components |
| `evaluation_plots.png` | Predicted vs actual + residual distribution |
| `rmse_by_zone.png` | Error breakdown by early/mid/late life zone |
| `results.json` | Final metrics (RMSE, MAE, R², NASA score) |

---

## 🔬 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)

- Source: NASA Ames Prognostics Data Repository
- Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- 4 sub-datasets (FD001–FD004) with varying fault modes and operating conditions
- Each engine runs from healthy state to failure, generating time-series sensor data

---

## 📚 References

1. Saxena A., Goebel K. et al. — *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008
2. Raissi M., Perdikaris P., Karniadakis G.E. — *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs*, Journal of Computational Physics, 2019
3. Li X. et al. — *Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks*, Reliability Engineering & System Safety, 2018

---

## 📄 License

This project is submitted as a course assignment. Dataset is publicly available from NASA and Kaggle.
