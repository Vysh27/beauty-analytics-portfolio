"""
Generate synthetic A/B experiment data for a beauty e-commerce platform.

Experiment: a redesigned product-detail page (PDP). Users are randomised 50/50
into control (old PDP) and treatment (new PDP). We log whether each user
converted and, if so, their order value. The treatment has a genuine but modest
effect, so the analysis has something real to detect.

Output: data/experiment.csv (user_id, variant, converted, order_value_inr, device).

Usage:
    python generate_data.py [--users 40000] [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# True effects baked into the data (the analysis should recover these).
BASE_CONV = 0.052          # control conversion rate
TREAT_UPLIFT = 0.0065      # +0.65pp absolute (~12.5% relative) — the real effect
BASE_AOV = 1450            # control average order value (INR)
TREAT_AOV_UPLIFT = 20      # small, not significant AOV effect


def generate(users: int = 40_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variant = rng.choice(["control", "treatment"], size=users, p=[0.5, 0.5])
    device = rng.choice(["mobile", "desktop"], size=users, p=[0.72, 0.28])

    conv_rate = np.where(variant == "treatment", BASE_CONV + TREAT_UPLIFT, BASE_CONV)
    # mobile converts a little lower — adds realistic heterogeneity
    conv_rate = conv_rate * np.where(device == "mobile", 0.92, 1.15)
    converted = rng.binomial(1, np.clip(conv_rate, 0, 1))

    aov_mean = np.where(variant == "treatment", BASE_AOV + TREAT_AOV_UPLIFT, BASE_AOV)
    order_value = np.where(
        converted == 1,
        np.clip(rng.normal(aov_mean, 450), 200, None).round(2),
        0.0,
    )

    df = pd.DataFrame({
        "user_id": [f"U{500000 + i}" for i in range(users)],
        "variant": variant,
        "device": device,
        "converted": converted,
        "order_value_inr": order_value,
    })
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = generate(args.users, args.seed)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "experiment.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} users to {path}")
    print(df.groupby("variant")["converted"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
