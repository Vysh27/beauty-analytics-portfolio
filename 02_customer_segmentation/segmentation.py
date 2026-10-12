"""
RFM customer segmentation with K-means for a beauty e-commerce platform.

Builds Recency / Frequency / Monetary features from the transaction log, chooses
k by silhouette score, clusters, and labels each cluster with a business-readable
persona. Exports segment profiles and a per-customer assignment the CRM team can use.

Pipeline:
    1. Aggregate transactions -> RFM per customer.
    2. Log-transform + standardise (RFM is heavily skewed).
    3. Pick k in [3..7] by silhouette, fit K-means.
    4. Profile clusters, map to personas by RFM ranking.
    5. Export segment_profiles.csv, customer_segments.csv, charts.

Usage:
    python segmentation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
OUT = HERE / "outputs"
TODAY = pd.Timestamp("2026-10-01")
PLOT_BG = "#FbFaF9"
PALETTE = ["#C2185B", "#6A7B8C", "#E59866", "#5B8A72", "#8E7CC3", "#4F6D7A", "#B5651D"]


def load_rfm():
    path = HERE / "data" / "transactions.csv"
    if not path.exists():
        raise SystemExit("data/transactions.csv not found — run `python generate_data.py` first.")
    tx = pd.read_csv(path, parse_dates=["order_date"])
    rfm = tx.groupby("customer_id").agg(
        recency=("order_date", lambda s: (TODAY - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("order_value_inr", "sum"),
    ).reset_index()
    return rfm


def choose_k(X, k_range=range(3, 8)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def label_segments(profile: pd.DataFrame) -> dict:
    """Map each cluster to a distinct persona from its R/F/M standing.

    Recency splits the clusters into 'recent' vs 'lapsed'; within each side we
    rank by value (frequency + monetary) and hand out the matching persona in
    order, so every cluster gets a unique, business-readable name.
    """
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    value = z(profile["frequency"]) + z(profile["monetary"])
    median_recency = profile["recency"].median()
    recent = profile["recency"] <= median_recency

    recent_names = ["Champions", "Loyal", "New / Promising"]
    lapsed_names = ["At Risk (high value)", "Hibernating", "Lost / low value"]

    labels = {}
    # recent clusters: best value first
    for rank, cid in enumerate(value[recent].sort_values(ascending=False).index):
        labels[cid] = recent_names[min(rank, len(recent_names) - 1)]
    # lapsed clusters: highest value is the one worth winning back
    for rank, cid in enumerate(value[~recent].sort_values(ascending=False).index):
        labels[cid] = lapsed_names[min(rank, len(lapsed_names) - 1)]
    return labels


def main():
    OUT.mkdir(exist_ok=True)
    rfm = load_rfm()
    print(f"Built RFM for {len(rfm):,} customers")

    feats = rfm[["recency", "frequency", "monetary"]].copy()
    feats_log = np.log1p(feats)                       # tame the skew
    X = StandardScaler().fit_transform(feats_log)

    best_k, scores = choose_k(X)
    print("Silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
    print(f"Chosen k = {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    rfm["cluster"] = km.labels_

    profile = rfm.groupby("cluster").agg(
        customers=("customer_id", "count"),
        recency=("recency", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
    )
    profile["segment"] = pd.Series(label_segments(profile))
    rfm["segment"] = rfm["cluster"].map(profile["segment"])
    profile["revenue_share"] = (
        rfm.groupby("cluster")["monetary"].sum() / rfm["monetary"].sum()
    ).round(3)

    profile_sorted = profile.sort_values("monetary", ascending=False)
    profile_sorted.to_csv(OUT / "segment_profiles.csv")
    rfm[["customer_id", "recency", "frequency", "monetary", "segment"]].to_csv(
        OUT / "customer_segments.csv", index=False
    )

    print("\nSegment profiles (median R/F/M):")
    print(profile_sorted[["segment", "customers", "recency", "frequency",
                          "monetary", "revenue_share"]].to_string(index=False))

    # ---- Chart: segment sizes vs revenue share ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    seg = profile_sorted.set_index("segment")
    x = np.arange(len(seg))
    w = 0.38
    cust_share = seg["customers"] / seg["customers"].sum()
    ax.bar(x - w / 2, cust_share, w, label="Share of customers", color=PALETTE[1])
    ax.bar(x + w / 2, seg["revenue_share"], w, label="Share of revenue", color=PALETTE[0])
    ax.set_xticks(x, seg.index, rotation=25, ha="right")
    ax.set_ylabel("Share")
    ax.set_title("Customers vs revenue by segment", fontweight="bold")
    ax.legend(frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "segment_value.png", dpi=130)
    plt.close(fig)

    # ---- Chart: R vs F scatter coloured by segment ------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    for i, (seg_name, grp) in enumerate(rfm.groupby("segment")):
        ax.scatter(grp["recency"], grp["frequency"], s=10, alpha=0.5,
                   color=PALETTE[i % len(PALETTE)], label=seg_name)
    ax.set_xlabel("Recency (days since last order)")
    ax.set_ylabel("Frequency (orders)")
    ax.set_title("Segments in Recency–Frequency space", fontweight="bold")
    ax.legend(frameon=False, markerscale=2, fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "segment_scatter.png", dpi=130)
    plt.close(fig)

    print(f"\nOutputs written to {OUT}/")


if __name__ == "__main__":
    main()
