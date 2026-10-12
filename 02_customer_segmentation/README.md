# 02 — Customer Segmentation (RFM + K-means)

Group customers into actionable segments so marketing can tailor treatment —
reward Champions, nurture Loyal customers, win back high-value At-Risk ones, and
not over-spend on the rest.

## Problem

A flat customer base hides very different behaviours. The business wants a small
number of clear, named segments it can build CRM journeys around.

## Approach

1. **Data** — `generate_data.py` creates a transaction log (~36k orders, 6,000
   customers) drawn from latent behaviour types, so there is real structure to
   recover.
2. **Features** — aggregate to **Recency, Frequency, Monetary** per customer.
3. **Model** — log-transform and standardise RFM (it is heavily skewed), then
   **K-means**; `k` is chosen by **silhouette score** over k = 3…7.
4. **Label** — clusters are mapped to business personas from their R/F/M
   standing, so each gets a distinct, readable name.

## Results (synthetic, seed 42)

k = 4 selected. Segment profiles (median R/F/M):

| Segment | Customers | Recency (d) | Frequency | Monetary (INR) | Revenue share |
|---------|-----------|-------------|-----------|----------------|---------------|
| Champions | 1,124 | 17 | 13 | 31,899 | **55.6%** |
| Loyal | 1,922 | 53 | 7 | 8,955 | 28.1% |
| At Risk (high value) | 1,172 | 317 | 4 | 5,833 | 11.8% |
| Hibernating | 1,782 | 314 | 1 | 1,509 | 4.5% |

The headline: **Champions are 19% of customers but 56% of revenue**, and the
*At Risk (high value)* segment is the clear win-back priority.

## Run it

```bash
python generate_data.py     # -> data/transactions.csv
python segmentation.py      # -> outputs/
```

## Outputs

| File | What it is |
|------|------------|
| `outputs/segment_profiles.csv` | Median R/F/M, size and revenue share per segment |
| `outputs/customer_segments.csv` | Per-customer segment assignment for the CRM |
| `outputs/segment_value.png` | Customers vs revenue share by segment |
| `outputs/segment_scatter.png` | Segments in Recency–Frequency space |

## Actioning the segments

Champions → loyalty perks and early sale access. Loyal → cross-sell to raise
frequency. At Risk (high value) → targeted win-back with a real incentive.
Hibernating → low-cost reactivation only.
