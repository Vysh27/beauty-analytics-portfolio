"""
A/B test analysis for a beauty e-commerce PDP redesign.

Covers the full experiment-analysis workflow an analyst is expected to own:
    - A sample-ratio-mismatch (SRM) check that the randomisation held.
    - Primary metric (conversion): two-proportion z-test, absolute & relative
      lift, 95% CI on the difference.
    - Secondary metric (AOV among converters): Welch's t-test + Mann-Whitney.
    - Revenue-per-user: the combined effect that actually matters.
    - Power / minimum-detectable-effect context and a segment (device) cut,
      with a note on why per-segment peeking inflates false positives.

Usage:
    python ab_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import (
    proportion_confint,
    proportions_ztest,
    proportion_effectsize,
)
from statsmodels.stats.power import NormalIndPower

HERE = Path(__file__).parent
OUT = HERE / "outputs"
PLOT_BG = "#FbFaF9"
ACCENT = "#C2185B"
ACCENT2 = "#6A7B8C"
ALPHA = 0.05


def load():
    path = HERE / "data" / "experiment.csv"
    if not path.exists():
        raise SystemExit("data/experiment.csv not found — run `python generate_data.py` first.")
    return pd.read_csv(path)


def srm_check(df):
    counts = df["variant"].value_counts()
    chi2, p = stats.chisquare(counts.values, [len(df) / 2] * 2)
    verdict = "OK (balanced)" if p > 0.01 else "WARNING: imbalance"
    print(f"SRM check: control={counts.get('control', 0):,} "
          f"treatment={counts.get('treatment', 0):,} | p={p:.3f} -> {verdict}")
    return p


def conversion_test(df):
    g = df.groupby("variant")
    conv = g["converted"].sum()
    n = g["converted"].count()
    c_rate, t_rate = conv["control"] / n["control"], conv["treatment"] / n["treatment"]

    stat, p = proportions_ztest([conv["treatment"], conv["control"]],
                                [n["treatment"], n["control"]])
    # 95% CI on the absolute difference (Newcombe via independent intervals)
    diff = t_rate - c_rate
    se = np.sqrt(c_rate * (1 - c_rate) / n["control"] +
                 t_rate * (1 - t_rate) / n["treatment"])
    ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se

    print("\n--- Primary metric: conversion rate ---")
    print(f"Control:   {c_rate:.4f}  ({conv['control']:,}/{n['control']:,})")
    print(f"Treatment: {t_rate:.4f}  ({conv['treatment']:,}/{n['treatment']:,})")
    print(f"Absolute lift: {diff*100:+.3f} pp   Relative lift: {diff/c_rate*100:+.2f}%")
    print(f"95% CI (abs): [{ci_lo*100:+.3f}, {ci_hi*100:+.3f}] pp")
    print(f"z = {stat:.3f},  p = {p:.4f}  -> "
          f"{'SIGNIFICANT' if p < ALPHA else 'not significant'} at a={ALPHA}")
    return {"c_rate": c_rate, "t_rate": t_rate, "diff": diff,
            "ci": (ci_lo, ci_hi), "p": p,
            "n_c": int(n["control"]), "n_t": int(n["treatment"])}


def aov_test(df):
    conv = df[df["converted"] == 1]
    c = conv[conv["variant"] == "control"]["order_value_inr"]
    t = conv[conv["variant"] == "treatment"]["order_value_inr"]
    t_stat, p_t = stats.ttest_ind(t, c, equal_var=False)       # Welch
    u_stat, p_u = stats.mannwhitneyu(t, c, alternative="two-sided")
    print("\n--- Secondary metric: AOV among converters ---")
    print(f"Control mean:   INR {c.mean():,.0f}  (n={len(c):,})")
    print(f"Treatment mean: INR {t.mean():,.0f}  (n={len(t):,})")
    print(f"Welch t-test p = {p_t:.3f} | Mann-Whitney p = {p_u:.3f} -> "
          f"{'significant' if p_t < ALPHA else 'not significant'}")
    return {"c_mean": c.mean(), "t_mean": t.mean(), "p": p_t}


def rpu_test(df):
    c = df[df["variant"] == "control"]["order_value_inr"]
    t = df[df["variant"] == "treatment"]["order_value_inr"]
    t_stat, p = stats.ttest_ind(t, c, equal_var=False)
    print("\n--- Combined: revenue per user ---")
    print(f"Control:   INR {c.mean():,.1f} / user")
    print(f"Treatment: INR {t.mean():,.1f} / user  "
          f"({(t.mean()/c.mean()-1)*100:+.2f}%)")
    print(f"Welch t-test p = {p:.4f} -> "
          f"{'significant' if p < ALPHA else 'not significant'}")
    return {"c": c.mean(), "t": t.mean(), "p": p}


def power_context(conv):
    eff = proportion_effectsize(conv["t_rate"], conv["c_rate"])
    achieved = NormalIndPower().solve_power(
        effect_size=eff, nobs1=conv["n_c"], alpha=ALPHA, ratio=conv["n_t"] / conv["n_c"])
    # sample size needed per arm for 80% power at this effect
    need = NormalIndPower().solve_power(effect_size=eff, power=0.8, alpha=ALPHA, ratio=1.0)
    print("\n--- Power ---")
    print(f"Achieved power at observed effect: {achieved:.2f}")
    print(f"Per-arm n for 80% power at this effect: {need:,.0f} "
          f"(had {conv['n_c']:,})")


def segment_cut(df):
    print("\n--- Segment cut: conversion by device (directional only) ---")
    rows = []
    for dev, grp in df.groupby("device"):
        g = grp.groupby("variant")["converted"]
        cr, tr = g.mean().get("control", np.nan), g.mean().get("treatment", np.nan)
        rows.append({"device": dev, "control": round(cr, 4), "treatment": round(tr, 4),
                     "rel_lift_%": round((tr / cr - 1) * 100, 1)})
        print(f"  {dev:<8} control {cr:.4f}  treatment {tr:.4f}  "
              f"lift {(tr/cr-1)*100:+.1f}%")
    print("  Note: per-segment tests multiply false-positive risk; treat as "
          "hypotheses, not decisions, unless pre-registered and corrected.")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    print(f"Loaded {len(df):,} users")

    srm_check(df)
    conv = conversion_test(df)
    aov = aov_test(df)
    rpu = rpu_test(df)
    power_context(conv)
    seg = segment_cut(df)

    # ---- exports ----------------------------------------------------------------
    pd.DataFrame([{
        "metric": "conversion", "control": round(conv["c_rate"], 4),
        "treatment": round(conv["t_rate"], 4),
        "rel_lift_pct": round(conv["diff"] / conv["c_rate"] * 100, 2),
        "p_value": round(conv["p"], 4),
        "significant": conv["p"] < ALPHA,
    }, {
        "metric": "revenue_per_user", "control": round(rpu["c"], 1),
        "treatment": round(rpu["t"], 1),
        "rel_lift_pct": round((rpu["t"] / rpu["c"] - 1) * 100, 2),
        "p_value": round(rpu["p"], 4), "significant": rpu["p"] < ALPHA,
    }]).to_csv(OUT / "results_summary.csv", index=False)
    seg.to_csv(OUT / "segment_breakdown.csv", index=False)

    # ---- chart: conversion with CI ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    rates = [conv["c_rate"], conv["t_rate"]]
    errs = [1.96 * np.sqrt(r * (1 - r) / n)
            for r, n in zip(rates, [conv["n_c"], conv["n_t"]])]
    ax.bar(["Control", "Treatment"], rates, yerr=errs, capsize=8,
           color=[ACCENT2, ACCENT], width=0.55)
    for i, r in enumerate(rates):
        ax.text(i, r + errs[i] + 0.0008, f"{r*100:.2f}%", ha="center", fontweight="bold")
    verdict = "significant" if conv["p"] < ALPHA else "not significant"
    ax.set_title(f"PDP redesign — conversion rate\n"
                 f"{conv['diff']/conv['c_rate']*100:+.1f}% relative, p={conv['p']:.3f} ({verdict})",
                 fontweight="bold")
    ax.set_ylabel("Conversion rate")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "conversion_result.png", dpi=130)
    plt.close(fig)

    print(f"\nOutputs written to {OUT}/")


if __name__ == "__main__":
    main()
