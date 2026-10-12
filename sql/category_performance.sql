-- ============================================================================
-- Category performance & MIS queries — beauty e-commerce
-- Dialect: ANSI SQL (tested patterns for Redshift / Databricks SQL / BigQuery;
--          minor function-name tweaks noted inline).
--
-- Demonstrates the SQL the JD asks for: complex joins, aggregation, window
-- functions, CTEs, cohort/retention logic and a reusable reporting view.
-- Assumed schema:
--   orders(order_id, customer_id, order_ts, category, units, gross_revenue,
--          discount_amount, channel)
--   customers(customer_id, signup_date, city, loyalty_tier)
-- ============================================================================


-- 1) DAILY CATEGORY TRACKER ---------------------------------------------------
-- Day-over-day and same-day-last-week comparison per category, the backbone of
-- a recurring MIS / category scorecard.
WITH daily AS (
    SELECT
        CAST(order_ts AS DATE)           AS order_date,
        category,
        SUM(units)                       AS units,
        SUM(gross_revenue)               AS revenue,
        SUM(discount_amount)             AS discount,
        COUNT(DISTINCT order_id)         AS orders,
        COUNT(DISTINCT customer_id)      AS buyers
    FROM orders
    WHERE order_ts >= DATEADD(day, -90, CURRENT_DATE)   -- BigQuery: DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    GROUP BY 1, 2
)
SELECT
    order_date,
    category,
    revenue,
    orders,
    buyers,
    ROUND(revenue * 1.0 / NULLIF(orders, 0), 2)                       AS aov,
    ROUND(discount * 100.0 / NULLIF(revenue + discount, 0), 1)        AS discount_pct,
    LAG(revenue, 1) OVER (PARTITION BY category ORDER BY order_date)  AS revenue_prev_day,
    LAG(revenue, 7) OVER (PARTITION BY category ORDER BY order_date)  AS revenue_same_day_lw,
    ROUND( (revenue - LAG(revenue, 7) OVER (PARTITION BY category ORDER BY order_date))
           * 100.0 / NULLIF(LAG(revenue, 7) OVER (PARTITION BY category ORDER BY order_date), 0), 1)
                                                                      AS wow_growth_pct
FROM daily
ORDER BY category, order_date;


-- 2) HOURLY SALES REPORT (sale-day pacing) ------------------------------------
-- Cumulative revenue by hour vs the same hour on the previous sale event —
-- the view a team watches live during a Pink Sale.
SELECT
    DATE_TRUNC('hour', order_ts)                                   AS hour_bucket,
    SUM(gross_revenue)                                             AS hourly_revenue,
    SUM(SUM(gross_revenue)) OVER (
        PARTITION BY CAST(order_ts AS DATE)
        ORDER BY DATE_TRUNC('hour', order_ts)
        ROWS UNBOUNDED PRECEDING)                                  AS cumulative_revenue
FROM orders
WHERE CAST(order_ts AS DATE) = CURRENT_DATE
GROUP BY 1
ORDER BY 1;


-- 3) CATEGORY CONTRIBUTION & RANK (month to date) -----------------------------
-- Each category's share of revenue and its rank, with running cumulative share
-- to see how concentrated sales are (a Pareto / 80-20 read).
WITH mtd AS (
    SELECT category, SUM(gross_revenue) AS revenue
    FROM orders
    WHERE order_ts >= DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY 1
)
SELECT
    category,
    revenue,
    RANK()        OVER (ORDER BY revenue DESC)                        AS revenue_rank,
    ROUND(revenue * 100.0 / SUM(revenue) OVER (), 1)                  AS pct_of_total,
    ROUND(SUM(revenue) OVER (ORDER BY revenue DESC
          ROWS UNBOUNDED PRECEDING) * 100.0 / SUM(revenue) OVER (), 1) AS cumulative_pct
FROM mtd
ORDER BY revenue DESC;


-- 4) NEW-CUSTOMER MONTHLY RETENTION COHORTS -----------------------------------
-- For each signup-month cohort, the share still purchasing N months later.
WITH first_month AS (
    SELECT customer_id, DATE_TRUNC('month', signup_date) AS cohort_month
    FROM customers
),
activity AS (
    SELECT DISTINCT o.customer_id,
           DATE_TRUNC('month', o.order_ts) AS active_month
    FROM orders o
),
cohort AS (
    SELECT
        f.cohort_month,
        DATEDIFF('month', f.cohort_month, a.active_month) AS month_offset,  -- BQ: DATE_DIFF(..., MONTH)
        COUNT(DISTINCT a.customer_id)                     AS active_customers
    FROM first_month f
    JOIN activity a ON a.customer_id = f.customer_id
    WHERE a.active_month >= f.cohort_month
    GROUP BY 1, 2
),
sizes AS (
    SELECT cohort_month, active_customers AS cohort_size
    FROM cohort WHERE month_offset = 0
)
SELECT
    c.cohort_month,
    c.month_offset,
    c.active_customers,
    s.cohort_size,
    ROUND(c.active_customers * 100.0 / NULLIF(s.cohort_size, 0), 1) AS retention_pct
FROM cohort c
JOIN sizes s USING (cohort_month)
WHERE c.month_offset <= 6
ORDER BY c.cohort_month, c.month_offset;


-- 5) REUSABLE VIEW: category daily scorecard ----------------------------------
-- Wrap the tracker as a view so dashboards and scheduled reports read one
-- governed definition instead of each re-implementing the logic.
CREATE OR REPLACE VIEW vw_category_daily_scorecard AS
SELECT
    CAST(order_ts AS DATE)                                   AS order_date,
    category,
    SUM(units)                                               AS units,
    SUM(gross_revenue)                                       AS revenue,
    COUNT(DISTINCT order_id)                                 AS orders,
    COUNT(DISTINCT customer_id)                              AS buyers,
    ROUND(SUM(gross_revenue) * 1.0 / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS aov,
    ROUND(SUM(discount_amount) * 100.0
          / NULLIF(SUM(gross_revenue) + SUM(discount_amount), 0), 1)         AS discount_pct
FROM orders
GROUP BY 1, 2;
