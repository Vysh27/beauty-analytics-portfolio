# SQL — Analytics & Reporting Queries

Production-style SQL for the recurring reporting a business-analytics team owns,
written against a simple beauty e-commerce schema (`orders`, `customers`).

`category_performance.sql` contains five worked queries:

1. **Daily category tracker** — revenue, orders, AOV, discount %, with
   day-over-day and same-day-last-week comparisons via `LAG()`. The backbone of
   a recurring MIS / category scorecard.
2. **Hourly sales report** — cumulative intraday revenue using a windowed running
   total, the kind of live pacing view watched during a sale event.
3. **Category contribution & rank** — each category's revenue share, `RANK()`,
   and running cumulative share for a Pareto read.
4. **New-customer retention cohorts** — monthly signup cohorts and the share
   still active N months later, built with CTEs and a self-join.
5. **Reusable scorecard view** — the tracker wrapped as
   `vw_category_daily_scorecard` so dashboards and scheduled reports read one
   governed definition.

## Dialect

Written in ANSI-style SQL that runs on Redshift / Databricks SQL with minor
tweaks; BigQuery equivalents for date functions are noted inline (e.g.
`DATEADD` → `DATE_SUB`, `DATEDIFF` → `DATE_DIFF`).

These are illustrative queries against an assumed schema — adjust table and
column names to your warehouse.
