"""
Generate a synthetic customer dataset for a beauty e-commerce platform.

The data is fabricated but follows realistic relationships: churn is driven by
recency, engagement decay, order frequency and discount dependence, with noise
on top so no single feature is a giveaway. Output is written to data/customers.csv.

Usage:
    python generate_data.py [--n 20000] [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["Skincare", "Makeup", "Haircare", "Fragrance", "Wellness", "Men"]


def generate(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Tenure on the platform, in days (how long they have been a customer).
    tenure_days = rng.integers(30, 1_460, size=n)

    # Order frequency: orders per active month, right-skewed.
    orders_per_month = np.clip(rng.gamma(shape=1.6, scale=0.9, size=n), 0.1, 12)
    total_orders = np.clip((orders_per_month * tenure_days / 30).round(), 1, None).astype(int)

    # Average order value (INR), varies by how premium the customer skews.
    premium_skew = rng.beta(2, 5, size=n)
    avg_order_value = (600 + premium_skew * 4_200 + rng.normal(0, 250, size=n)).clip(200, 8_000)

    monetary = (total_orders * avg_order_value).round(2)

    # Engagement signals.
    days_since_last_order = rng.integers(1, 365, size=n)
    app_opens_last_30d = rng.poisson(lam=np.clip(8 - days_since_last_order / 40, 0.2, 8), size=n)
    sessions_last_30d = (app_opens_last_30d * rng.uniform(0.6, 1.4, size=n)).round().astype(int)

    # Breadth of catalogue the customer buys across (1..6 categories).
    category_diversity = np.clip(
        rng.binomial(len(CATEGORIES), 0.25 + premium_skew * 0.3, size=n), 1, len(CATEGORIES)
    )

    # Discount dependence: share of orders placed on promotion (0..1).
    discount_rate = rng.beta(2, 3, size=n).round(3)

    # Service friction.
    support_tickets = rng.poisson(lam=0.4, size=n)
    returns = rng.poisson(lam=total_orders * 0.05).clip(0, total_orders)
    return_rate = (returns / total_orders).round(3)

    is_newsletter_sub = rng.binomial(1, 0.55, size=n)
    loyalty_tier = rng.choice(["None", "Silver", "Gold", "Platinum"], size=n, p=[0.45, 0.30, 0.18, 0.07])
    tier_score = pd.Series(loyalty_tier).map({"None": 0, "Silver": 1, "Gold": 2, "Platinum": 3}).to_numpy()

    # ---- Latent churn propensity -------------------------------------------------
    # Standardise drivers, then combine into a logit. Positive weight => more churn.
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    logit = (
        1.35 * z(days_since_last_order)        # the dominant driver
        - 0.80 * z(sessions_last_30d)          # engagement protects
        - 0.55 * z(orders_per_month)           # habit protects
        + 0.45 * z(discount_rate)              # discount-only buyers churn more
        - 0.40 * z(category_diversity)         # breadth protects
        - 0.35 * z(tier_score)                 # loyalty tier protects
        + 0.30 * z(return_rate)
        + 0.25 * z(support_tickets)
        - 0.20 * z(np.log1p(monetary))
        - 0.9                                   # base rate shift
        + rng.normal(0, 0.8, size=n)           # irreducible noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{100000 + i}" for i in range(n)],
            "tenure_days": tenure_days,
            "total_orders": total_orders,
            "orders_per_month": orders_per_month.round(2),
            "avg_order_value_inr": avg_order_value.round(2),
            "monetary_inr": monetary,
            "days_since_last_order": days_since_last_order,
            "app_opens_last_30d": app_opens_last_30d,
            "sessions_last_30d": sessions_last_30d,
            "category_diversity": category_diversity,
            "discount_order_rate": discount_rate,
            "return_rate": return_rate,
            "support_tickets": support_tickets,
            "is_newsletter_sub": is_newsletter_sub,
            "loyalty_tier": loyalty_tier,
            "churned": churned,
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.n, args.seed)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "customers.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows to {path}")
    print(f"Overall churn rate: {df['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
