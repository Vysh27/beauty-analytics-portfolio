"""
Daily demand forecasting for a beauty e-commerce category.

Fits seasonal models to the historical units series and forecasts the next 60
days with a confidence interval, benchmarked against a seasonal-naive baseline.
Reports MAPE / RMSE on a held-out test window so the accuracy claim is honest.

Models:
    - Seasonal-naive baseline (last week repeated) — the bar to beat.
    - Holt-Winters (additive trend + weekly seasonality).
    - SARIMA (auto-ish fixed order with weekly seasonality).

Usage:
    python forecast.py [--category Skincare] [--horizon 60]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "outputs"
PLOT_BG = "#FbFaF9"
ACCENT = "#C2185B"
ACCENT2 = "#6A7B8C"
TEST_DAYS = 60


def load_series(category: str) -> pd.Series:
    path = HERE / "data" / "daily_sales.csv"
    if not path.exists():
        raise SystemExit("data/daily_sales.csv not found — run `python generate_data.py` first.")
    df = pd.read_csv(path, parse_dates=["date"])
    s = (df[df["category"] == category]
         .set_index("date")["units"].asfreq("D"))
    if s.empty:
        raise SystemExit(f"No data for category '{category}'. "
                         f"Options: {sorted(df['category'].unique())}")
    return s


def mape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Skincare")
    ap.add_argument("--horizon", type=int, default=60)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    s = load_series(args.category)
    train, test = s.iloc[:-TEST_DAYS], s.iloc[-TEST_DAYS:]
    print(f"{args.category}: {len(s)} days | train {len(train)} / test {len(test)}")

    results = {}

    # 1) Seasonal-naive baseline: repeat the last observed week.
    last_week = train.iloc[-7:].to_numpy()
    naive = np.resize(last_week, len(test))
    results["Seasonal-naive"] = {"pred": pd.Series(naive, index=test.index)}

    # 2) Holt-Winters
    hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                              seasonal_periods=7).fit()
    results["Holt-Winters"] = {"pred": hw.forecast(len(test))}

    # 3) SARIMA with weekly seasonality
    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    results["SARIMA"] = {"pred": sarima.predict(start=test.index[0], end=test.index[-1])}

    print("\nBack-test accuracy (lower is better):")
    print(f"{'model':<16}{'MAPE %':>9}{'RMSE':>10}")
    for name, r in results.items():
        r["mape"], r["rmse"] = mape(test, r["pred"]), rmse(test, r["pred"])
        print(f"{name:<16}{r['mape']:>9.2f}{r['rmse']:>10.1f}")

    best_name = min(results, key=lambda k: results[k]["mape"])
    print(f"\nBest model: {best_name} (MAPE {results[best_name]['mape']:.2f}%)")

    pd.DataFrame(
        [{"model": n, "mape_pct": round(r["mape"], 2), "rmse": round(r["rmse"], 1)}
         for n, r in results.items()]
    ).to_csv(OUT / f"{args.category.lower()}_accuracy.csv", index=False)

    # ---- Refit best model on full history, forecast forward ---------------------
    if best_name == "Holt-Winters":
        final = ExponentialSmoothing(s, trend="add", seasonal="add",
                                     seasonal_periods=7).fit()
        fc = final.forecast(args.horizon)
        ci = None
    elif best_name == "SARIMA":
        final = SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = final.get_forecast(args.horizon)
        fc = pred.predicted_mean
        ci = pred.conf_int(alpha=0.2)
    else:
        future_idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1),
                                   periods=args.horizon, freq="D")
        fc = pd.Series(np.resize(s.iloc[-7:].to_numpy(), args.horizon), index=future_idx)
        ci = None

    fc_df = fc.rename("forecast_units").to_frame()
    fc_df.to_csv(OUT / f"{args.category.lower()}_forecast.csv")

    # ---- Chart ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    s.iloc[-180:].plot(ax=ax, color=ACCENT2, lw=1, label="Actual (last 180d)")
    fc.plot(ax=ax, color=ACCENT, lw=2, label=f"Forecast ({best_name}, {args.horizon}d)")
    if ci is not None:
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                        color=ACCENT, alpha=0.15, label="80% interval")
    ax.set_title(f"{args.category} — daily units forecast", fontweight="bold")
    ax.set_ylabel("Units/day")
    ax.legend(frameon=False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{args.category.lower()}_forecast.png", dpi=130)
    plt.close(fig)

    print(f"Outputs written to {OUT}/")


if __name__ == "__main__":
    main()
