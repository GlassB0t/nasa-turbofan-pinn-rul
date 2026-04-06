"""
main.py
──────────────────────────────────────────────────────────────────────────────
ONE-CLICK RUN SCRIPT
NASA C-MAPSS Turbofan Engine — RUL Prediction via Physics-Informed Neural Network
──────────────────────────────────────────────────────────────────────────────

Usage (from project root):
    python main.py

This script:
  1. Downloads / verifies the C-MAPSS dataset (or prompts you to place it)
  2. Runs the full PINN training pipeline on FD001
  3. Evaluates on the test set
  4. Saves all plots and metrics to results/
  5. Compares PINN results against a Random Forest baseline
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse

# ── Add src/ to path so imports work regardless of working directory ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from train    import train
from evaluate import compare_with_baseline


def check_data(data_dir: str) -> bool:
    """Verify the C-MAPSS data files exist."""
    required = [
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
        "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
        "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
        "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print("\n" + "=" * 60)
        print("  DATA FILES NOT FOUND")
        print("=" * 60)
        print(f"  Missing files in '{data_dir}':")
        for f in missing:
            print(f"    - {f}")
        print("""
  How to get the data:
  ────────────────────
  1. Visit Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
  2. Download and unzip the dataset
  3. Place all .txt files inside the 'data/' folder:
       data/train_FD001.txt
       data/test_FD001.txt
       data/RUL_FD001.txt
       ... (same for FD002, FD003, FD004)
  4. Re-run: python main.py
""")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="PINN-RUL: Physics-Informed RUL Prediction for Turbofan Engines"
    )
    parser.add_argument("--subset",         type=str,   default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data_dir",       type=str,   default="data/")
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch_size",     type=int,   default=512)
    parser.add_argument("--physics_warmup", type=int,   default=10,
                        help="Epoch after which physics losses activate")
    parser.add_argument("--save_dir",       type=str,   default="results/")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║   PINN-RUL: Physics-Informed Neural Network                  ║
║   Remaining Useful Life Prediction — Turbofan Engines        ║
║   NASA C-MAPSS Dataset                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Verify data
    if not check_data(args.data_dir):
        sys.exit(1)

    # Run training + evaluation
    metrics = train(args)

    # Print comparison
    compare_with_baseline(metrics)

    print("""
╔══════════════════════════════════════════════════════════════╗
║   RUN COMPLETE                                               ║
║   Check the 'results/' folder for:                           ║
║     • best_model.pth         — saved model weights           ║
║     • training_curves.png    — loss history                  ║
║     • evaluation_plots.png   — predicted vs actual           ║
║     • rmse_by_zone.png       — error breakdown by RUL zone   ║
║     • results.json           — all metrics                   ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
