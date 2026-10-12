"""
Generate ~2 years of daily category sales for a beauty e-commerce platform.

Each category series carries a trend, weekly seasonality, an annual festive lift,
and sharp spikes on sale events (incl. a recurring 'Pink Sale'), plus noise.
Output: data/daily_sales.csv  (date, category, units, revenue_inr, is_sale_event).

Usage:
    python generate_data.py [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2024-09-01")
END = pd.Timestamp("2026-09-30")

CATEGORIES = {
    #              base, trend/day, weekly_amp, price
    "Skincare":   (420, 0.25, 0.22, 1400),
    "Makeup":     (380, 0.18, 0.30, 1100),
    "Haircare":   (300, 0.12, 0.18, 900),
    "Fragrance":  (160, 0.10, 0.35, 2200),
}

# Recurring multi-day sale events (name -> list of (start, days, lift multiplier)).
SALE_EVENTS = [
    ("2024-11-01", 6, 2.6),   # festive / Diwali window
    ("2025-01-10", 4, 1.8),
    ("2025-03-07", 5, 2.2),   # Pink Sale
    ("2025-07-11", 5, 2.1),
    ("2025-11-01", 6, 2.8),
    ("2026-03-06", 5, 2.3),   # Pink Sale
    ("2026-07-10", 5, 2.2),
]


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    t = np.arange(len(dates))

    sale_flag = np.zeros(len(dates))
    sale_mult = np.ones(len(dates))
    for start, days, lift in SALE_EVENTS:
        s = pd.Timestamp(start)
        mask = (dates >= s) & (dates < s + pd.Timedelta(days=days))
        sale_flag[mask] = 1
        sale_mult[mask] = lift

    rows = []
    for cat, (base, trend, w_amp, price) in CATEGORIES.items():
        weekly = 1 + w_amp * np.sin(2 * np.pi * (dates.dayofweek) / 7)
        # annual festive bump around Oct-Nov
        annual = 1 + 0.25 * np.exp(-((dates.dayofyear - 305) ** 2) / (2 * 25 ** 2))
        level = (base + trend * t) * weekly * annual * sale_mult
        noise = rng.normal(1, 0.08, size=len(dates))
        units = np.clip(level * noise, 0, None).round().astype(int)
        revenue = (units * price * rng.normal(1, 0.03, size=len(dates))).round(2)
        rows.append(pd.DataFrame({
            "date": dates, "category": cat, "units": units,
            "revenue_inr": revenue, "is_sale_event": sale_flag.astype(int),
        }))

    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = generate(args.seed)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "daily_sales.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows ({df['category'].nunique()} categories, "
          f"{df['date'].nunique()} days) to {path}")


if __name__ == "__main__":
    main()
