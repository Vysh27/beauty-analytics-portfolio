"""
Beauty e-commerce sales dashboard (Streamlit).

A self-serve view over the synthetic order log: KPI tiles with period-on-period
deltas, a revenue trend, category and segment breakdowns, a channel split, and a
downloadable detail table — all driven by sidebar filters (date range, category,
channel).

Run:
    python generate_data.py      # writes data/orders.csv
    streamlit run app.py
"""
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
ACCENT = "#C2185B"
ACCENT2 = "#6A7B8C"
DATA = HERE / "data" / "orders.csv"

st.set_page_config(page_title="Beauty Sales Dashboard", page_icon="💄",
                   layout="wide")


# ---- data ------------------------------------------------------------------
@st.cache_data
def load_orders() -> pd.DataFrame:
    if not DATA.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA, parse_dates=["order_date"])
    return df


def kpis(df: pd.DataFrame) -> dict:
    """Headline metrics for a slice of orders."""
    return {
        "revenue": df["revenue_inr"].sum(),
        "orders": len(df),
        "aov": df["revenue_inr"].sum() / len(df) if len(df) else 0,
        "customers": df["customer_id"].nunique(),
    }


def pct_delta(curr: float, prev: float):
    if not prev:
        return None
    return (curr - prev) / prev * 100


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


# ---- charts ----------------------------------------------------------------
def revenue_trend(df: pd.DataFrame):
    wk = (df.set_index("order_date")
            .resample("W")["revenue_inr"].sum()
            .reset_index())
    return (
        alt.Chart(wk)
        .mark_line(point=False, color=ACCENT, strokeWidth=2.5)
        .encode(
            x=alt.X("order_date:T", title=None),
            y=alt.Y("revenue_inr:Q", title="Weekly revenue (₹)",
                    axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("order_date:T", title="Week"),
                     alt.Tooltip("revenue_inr:Q", title="Revenue", format=",.0f")],
        )
        .properties(height=300)
    )


def bar(df: pd.DataFrame, dim: str, title: str):
    g = (df.groupby(dim)["revenue_inr"].sum()
           .sort_values(ascending=False).reset_index())
    return (
        alt.Chart(g)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("revenue_inr:Q", title="Revenue (₹)", axis=alt.Axis(format="~s")),
            y=alt.Y(f"{dim}:N", title=None, sort="-x"),
            tooltip=[dim, alt.Tooltip("revenue_inr:Q", title="Revenue", format=",.0f")],
        )
        .properties(height=260, title=title)
    )


def channel_donut(df: pd.DataFrame):
    g = df.groupby("channel")["revenue_inr"].sum().reset_index()
    return (
        alt.Chart(g)
        .mark_arc(innerRadius=55)
        .encode(
            theta="revenue_inr:Q",
            color=alt.Color("channel:N",
                            scale=alt.Scale(range=[ACCENT, ACCENT2, "#E59866"]),
                            legend=alt.Legend(title="Channel")),
            tooltip=["channel", alt.Tooltip("revenue_inr:Q", title="Revenue", format=",.0f")],
        )
        .properties(height=260, title="Revenue by channel")
    )


# ---- app -------------------------------------------------------------------
def main():
    st.title("💄 Beauty E-commerce — Sales Dashboard")
    st.caption("Synthetic data for portfolio demonstration. Filters apply to every tile below.")

    df = load_orders()
    if df.empty:
        st.warning("No data found. Run `python generate_data.py` first to create data/orders.csv.")
        st.stop()

    # ---- sidebar filters ----
    st.sidebar.header("Filters")
    min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_d, max_d),
                                       min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d, max_d

    cats = sorted(df["category"].unique())
    chosen_cats = st.sidebar.multiselect("Category", cats, default=cats)
    chans = sorted(df["channel"].unique())
    chosen_chans = st.sidebar.multiselect("Channel", chans, default=chans)

    mask = (
        (df["order_date"].dt.date >= start_d)
        & (df["order_date"].dt.date <= end_d)
        & (df["category"].isin(chosen_cats))
        & (df["channel"].isin(chosen_chans))
    )
    view = df[mask]

    if view.empty:
        st.info("No orders match the current filters.")
        st.stop()

    # ---- previous equal-length period for deltas ----
    span = (end_d - start_d) + timedelta(days=1)
    prev_start, prev_end = start_d - span, start_d - timedelta(days=1)
    prev_mask = (
        (df["order_date"].dt.date >= prev_start)
        & (df["order_date"].dt.date <= prev_end)
        & (df["category"].isin(chosen_cats))
        & (df["channel"].isin(chosen_chans))
    )
    cur, prev = kpis(view), kpis(df[prev_mask])

    # ---- KPI row ----
    c1, c2, c3, c4 = st.columns(4)
    for col, label, key, fmt in [
        (c1, "Revenue", "revenue", inr),
        (c2, "Orders", "orders", lambda x: f"{x:,}"),
        (c3, "Avg order value", "aov", inr),
        (c4, "Unique customers", "customers", lambda x: f"{x:,}"),
    ]:
        d = pct_delta(cur[key], prev[key])
        col.metric(label, fmt(cur[key]),
                   None if d is None else f"{d:+.1f}% vs prev period")

    st.divider()

    # ---- trend ----
    st.subheader("Revenue trend")
    st.altair_chart(revenue_trend(view), use_container_width=True)

    # ---- breakdowns ----
    left, right = st.columns(2)
    with left:
        st.altair_chart(bar(view, "category", "Revenue by category"),
                        use_container_width=True)
    with right:
        st.altair_chart(bar(view, "segment", "Revenue by customer segment"),
                        use_container_width=True)

    st.altair_chart(channel_donut(view), use_container_width=True)

    # ---- detail table + download ----
    st.subheader("Order detail")
    table = (view.groupby(["category", "segment"])
                 .agg(orders=("order_id", "count"),
                      revenue=("revenue_inr", "sum"),
                      aov=("revenue_inr", "mean"))
                 .round(0).reset_index()
                 .sort_values("revenue", ascending=False))
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download filtered orders (CSV)",
                       view.to_csv(index=False).encode(),
                       file_name="filtered_orders.csv", mime="text/csv")


if __name__ == "__main__":
    main()
