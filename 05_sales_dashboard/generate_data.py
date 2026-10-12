"""
Generate a synthetic order-level dataset to power the sales dashboard.

One row per order over ~15 months, with category, channel, customer and an
assigned RFM-style segment, plus trend, weekly seasonality and sale-event spikes.
Output: data/orders.csv.

Usage:
    python generate_data.py [--orders 60000] [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-07-01")
END = pd.Timestamp("2026-09-30")

CATEGORIES = {   # category -> (relative popularity, mean price INR)
    "Skincare":  (0.28, 1400),
    "Makeup":    (0.24, 1100),
    "Haircare":  (0.18, 900),
    "Fragrance": (0.12, 2200),
    "Wellness":  (0.10, 1300),
    "Men":       (0.08, 1000),
}
CHANNELS = {"App": 0.55, "Web": 0.30, "Marketplace": 0.15}
SEGMENTS = {"Champions": 0.14, "Loyal": 0.22, "New / Promising": 0.20,
            "At Risk": 0.18, "Hibernating": 0.26}

# Multi-day sale windows that lift daily order volume.
SALE_EVENTS = [("2025-11-01", 6, 2.7), ("2026-03-06", 5, 2.3), ("2026-07-10", 5, 2.2)]


def _daily_weights(dates, rng):
    t = np.arange(len(dates))
    trend = 1 + 0.0007 * t                                   # slow growth
    weekly = 1 + 0.25 * np.sin(2 * np.pi * np.asarray(dates.dayofweek) / 7)
    festive = 1 + 0.30 * np.exp(-((np.asarray(dates.dayofyear) - 305) ** 2) / (2 * 25 ** 2))
    w = np.asarray(trend * weekly * festive, dtype=float)
    for start, days, lift in SALE_EVENTS:
        s = pd.Timestamp(start)
        w[(dates >= s) & (dates < s + pd.Timedelta(days=days))] *= lift
    return w / w.sum()


def generate(orders: int = 60_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    day_probs = _daily_weights(dates, rng)
    order_days = rng.choice(dates, size=orders, p=day_probs)

    cats = list(CATEGORIES)
    cat = rng.choice(cats, size=orders, p=[CATEGORIES[c][0] for c in cats])
    prices = np.array([CATEGORIES[c][1] for c in cat])

    channel = rng.choice(list(CHANNELS), size=orders, p=list(CHANNELS.values()))

    n_customers = max(2000, orders // 8)
    customer_id = rng.integers(0, n_customers, size=orders)
    # assign each customer a stable segment
    seg_pool = rng.choice(list(SEGMENTS), size=n_customers, p=list(SEGMENTS.values()))
    segment = seg_pool[customer_id]

    units = rng.integers(1, 4, size=orders)
    # champions/loyal spend a bit more per order
    seg_mult = pd.Series(segment).map(
        {"Champions": 1.35, "Loyal": 1.15, "New / Promising": 1.0,
         "At Risk": 0.95, "Hibernating": 0.85}).to_numpy()
    revenue = (units * prices * seg_mult * rng.normal(1, 0.12, orders)).clip(200).round(2)

    df = pd.DataFrame({
        "order_id": [f"ORD{1_000_000 + i}" for i in range(orders)],
        "order_date": pd.to_datetime(order_days),
        "customer_id": [f"C{200000 + c}" for c in customer_id],
        "category": cat,
        "channel": channel,
        "segment": segment,
        "units": units,
        "revenue_inr": revenue,
    }).sort_values("order_date").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = generate(args.orders, args.seed)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "orders.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} orders ({df['customer_id'].nunique():,} customers, "
          f"{df['order_date'].dt.date.nunique()} days) to {path}")
    print(f"Total revenue: INR {df['revenue_inr'].sum():,.0f}")


if __name__ == "__main__":
    main()
