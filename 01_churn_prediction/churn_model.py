"""
Customer churn prediction for a beauty e-commerce platform.

Trains and compares two models (regularised Logistic Regression and Gradient
Boosting) to predict which customers will churn, then scores the full base and
exports a ranked at-risk list the retention team can action.

Pipeline:
    1. Load data (run generate_data.py first).
    2. Preprocess: one-hot categoricals, scale numerics.
    3. Train/test split (stratified), fit both models.
    4. Evaluate: ROC-AUC, PR-AUC, precision/recall/F1 at a chosen threshold.
    5. Explain: permutation importance on the better model.
    6. Score everyone, write at_risk_customers.csv (top decile).

Usage:
    python churn_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
OUT = HERE / "outputs"
TARGET = "churned"
ID_COL = "customer_id"
CAT_COLS = ["loyalty_tier"]
PLOT_BG = "#FbFaF9"
ACCENT = "#C2185B"   # deep pink, on-theme for a beauty brand
ACCENT2 = "#6A7B8C"


def load():
    path = HERE / "data" / "customers.csv"
    if not path.exists():
        raise SystemExit("data/customers.csv not found — run `python generate_data.py` first.")
    return pd.read_csv(path)


def build_pipeline(model, num_cols):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipe, X_te, y_te, threshold=0.5):
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_te, proba)
    ap = average_precision_score(y_te, proba)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.3f}   PR-AUC: {ap:.3f}")
    print(classification_report(y_te, pred, digits=3))
    return {"name": name, "pipe": pipe, "proba": proba, "auc": auc, "ap": ap}


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    print(f"Loaded {len(df):,} customers | churn rate {df[TARGET].mean():.1%}")

    feature_cols = [c for c in df.columns if c not in (TARGET, ID_COL)]
    num_cols = [c for c in feature_cols if c not in CAT_COLS]
    X, y = df[feature_cols], df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    for name, est in models.items():
        pipe = build_pipeline(est, num_cols).fit(X_tr, y_tr)
        results.append(evaluate(name, pipe, X_te, y_te))

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest model by ROC-AUC: {best['name']} ({best['auc']:.3f})")

    # ---- ROC curve comparison ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    for r, c in zip(results, [ACCENT, ACCENT2]):
        fpr, tpr, _ = roc_curve(y_te, r["proba"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{r['name']} (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#B0A9A3", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Churn model — ROC curve", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=130)
    plt.close(fig)

    # ---- Confusion matrix (best model) ------------------------------------------
    pred_best = (best["proba"] >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred_best)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=PLOT_BG)
    ax.imshow(cm, cmap="PuRd")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "#333", fontweight="bold")
    ax.set_xticks([0, 1], ["Pred: stay", "Pred: churn"])
    ax.set_yticks([0, 1], ["Actual: stay", "Actual: churn"])
    ax.set_title(f"Confusion matrix — {best['name']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # ---- Permutation importance on the best model -------------------------------
    perm = permutation_importance(
        best["pipe"], X_te, y_te, n_repeats=8, random_state=42, scoring="roc_auc"
    )
    imp = (
        pd.Series(perm.importances_mean, index=feature_cols)
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    imp[::-1].plot.barh(ax=ax, color=ACCENT)
    ax.set_title("Top churn drivers (permutation importance)", fontweight="bold")
    ax.set_xlabel("Mean drop in ROC-AUC when shuffled")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=130)
    plt.close(fig)

    # ---- Score the whole base, export the actionable list -----------------------
    df["churn_score"] = best["pipe"].predict_proba(df[feature_cols])[:, 1]
    df["risk_band"] = pd.qcut(df["churn_score"], [0, 0.5, 0.8, 0.9, 1.0],
                              labels=["Low", "Medium", "High", "Critical"])
    at_risk = (
        df.sort_values("churn_score", ascending=False)
        .loc[df["risk_band"].isin(["High", "Critical"]),
             [ID_COL, "churn_score", "risk_band", "days_since_last_order",
              "monetary_inr", "loyalty_tier"]]
    )
    at_risk.to_csv(OUT / "at_risk_customers.csv", index=False)

    # ---- Summary metrics ---------------------------------------------------------
    summary = pd.DataFrame(
        [{"model": r["name"], "roc_auc": round(r["auc"], 4),
          "pr_auc": round(r["ap"], 4)} for r in results]
    )
    summary.to_csv(OUT / "model_metrics.csv", index=False)

    print(f"\nFlagged {len(at_risk):,} High/Critical-risk customers "
          f"({len(at_risk) / len(df):.1%} of base).")
    print(f"Outputs written to {OUT}/")


if __name__ == "__main__":
    main()
