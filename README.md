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
streamlit run app.py                   # opens at http://localhost:8501
```

The app generates its dataset automatically on first run (60k synthetic orders),
so it also works out of the box when deployed to Streamlit Community Cloud — no
manual data step needed. You can still run `python generate_data.py` yourself to
regenerate `data/orders.csv`.

### Deploy a public link (optional)

Push this repo to GitHub, go to [share.streamlit.io](https://share.streamlit.io),
connect the repo, set the main file path to `05_sales_dashboard/app.py`, and
deploy — you get a public `*.streamlit.app` URL you can share.

## No-install version (standalone HTML)

If you just want to *look* at the dashboard with no Python or install, open
`sales_dashboard.html` in any browser — it's fully self-contained (data embedded,
charts drawn with plain SVG, no internet needed) and has the same filters, KPI
tiles and breakdowns. Rebuild it anytime with `python build_html_dashboard.py`.

**View it live (clickable link):**
[open the dashboard](https://htmlpreview.github.io/?https://github.com/Vysh27/beauty-analytics-portfolio/blob/main/05_sales_dashboard/sales_dashboard.html)
— this renders the HTML straight from the repo (works once the file is uploaded).
For a cleaner permanent URL you can also enable GitHub Pages (Settings → Pages →
Deploy from branch `main`), which serves it at
`https://vysh27.github.io/beauty-analytics-portfolio/05_sales_dashboard/sales_dashboard.html`.

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
