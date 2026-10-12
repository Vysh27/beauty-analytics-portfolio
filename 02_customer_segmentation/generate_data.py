"""
Generate a synthetic transaction log for a beauty e-commerce platform.

One row per order. Customers are drawn from a few latent behaviour types
(champions, loyal, promo-hunters, one-timers, lapsing) so that RFM-based
clustering has real structure to recover. Output: data/transactions.csv.

Usage:
    python generate_data.py [--customers 6000] [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TODAY = pd.Timestamp("2026-10-01")
CATEGORIES = ["Skincare", "Makeup", "Haircare", "Fragrance", "Wellness", "Men"]

# latent types: (share, orders_lambda, recency_days_range, aov_mean)
TYPES = {
    "champion":    (0.14, 14, (1, 30),   2600),
    "loyal":       (0.22, 8,  (10, 75),  1700),
    "promo_hunter":(0.18, 6,  (20, 120), 950),
    "one_timer":   (0.26, 1,  (90, 540), 1300),
    "lapsing":     (0.20, 4,  (150, 540),1500),
}


def generate(n_customers: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    types = list(TYPES)
    probs = [TYPES[t][0] for t in types]

    for i in range(n_customers):
        t = rng.choice(types, p=probs)
        _, lam, (rmin, rmax), aov_mean = TYPES[t]
        cust = f"CUST{200000 + i}"

        n_orders = max(1, rng.poisson(lam))
        recency = rng.integers(rmin, rmax + 1)           # days since last order
        # spread orders back in time from the recency point
        last_order = TODAY - pd.Timedelta(days=int(recency))
        gaps = np.sort(rng.integers(5, 90, size=n_orders))
        order_dates = [last_order - pd.Timedelta(days=int(g)) for g in gaps[::-1]]
        order_dates[-1] = last_order

        for d in order_dates:
            aov = max(200, rng.normal(aov_mean, aov_mean * 0.3))
            rows.append(
                {
                    "customer_id": cust,
                    "order_date": d.date().isoformat(),
                    "order_value_inr": round(float(aov), 2),
                    "category": rng.choice(CATEGORIES),
                    "latent_type": t,  # kept only for optional validation; drop before modelling
                }
            )

    df = pd.DataFrame(rows).sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.customers, args.seed)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "transactions.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} orders for {df['customer_id'].nunique():,} customers to {path}")


if __name__ == "__main__":
    main()
