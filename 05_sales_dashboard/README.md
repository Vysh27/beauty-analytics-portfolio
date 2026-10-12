# 05 — Sales Dashboard (Streamlit)

An interactive, self-serve sales dashboard over the beauty e-commerce order log —
the kind of recurring reporting surface a business-analytics team owns, with
filters that let stakeholders answer their own questions instead of asking for a
new cut each time.

## What it shows

- **KPI tiles** — revenue, orders, average order value, unique customers, each
  with a **period-on-period delta** against the previous equal-length window.
- **Revenue trend** — weekly revenue over time (sale-event spikes visible).
- **Breakdowns** — revenue by category and by customer segment.
- **Channel split** — App / Web / Marketplace share.
- **Detail table** — category × segment summary, with a **download-to-CSV** button.

Everything responds to the sidebar filters: **date range, category, channel**.

## Run it

```bash
pip install -r ../requirements.txt     # needs streamlit + altair
python generate_data.py                # -> data/orders.csv (60k synthetic orders)
streamlit run app.py                   # opens at http://localhost:8501
```

## How it's built

- `generate_data.py` creates 60,000 synthetic orders over ~15 months with
  category, channel, customer and an assigned RFM-style segment, plus trend,
  weekly seasonality and sale-event spikes.
- `app.py` keeps data loading and aggregation in plain functions (cached with
  `st.cache_data`), so the logic is testable outside Streamlit; charts are built
  with Altair and themed to match the rest of the portfolio.

## Why it's here

Dashboarding and self-serve metrics are central to the Analyst II role. This
project shows the end-to-end: model the data, define the KPIs (including a
correct period-on-period comparison), and ship an interactive surface a
non-analyst can actually use.

## Notes

Synthetic data, so the numbers describe no real business. On real data this would
sit on a warehouse table (or the dbt models from the other projects) and refresh
on a schedule; the same layout ports directly to Tableau or Power BI.
