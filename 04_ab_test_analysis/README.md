# 04 — A/B Test Analysis

Analyse a controlled experiment end to end, the way an analyst is expected to own
it: validate the randomisation, test the primary and secondary metrics with the
right statistics, quantify the lift with confidence intervals, and state a clear,
caveated recommendation.

## Problem

Does a **redesigned product-detail page (PDP)** improve conversion? Users are
randomised 50/50 into control (old PDP) and treatment (new PDP). We want a
decision that stands up to scrutiny — not just a difference in averages.

## Approach

1. **Data** — `generate_data.py` simulates 40,000 users with a real but modest
   treatment effect baked in (~+0.65pp absolute conversion), plus device
   heterogeneity so segment cuts are meaningful.
2. **Guardrail** — a **sample-ratio-mismatch (SRM)** chi-square check that the
   split actually held before trusting any result.
3. **Primary metric (conversion)** — two-proportion **z-test**, absolute and
   relative lift, and a 95% CI on the difference.
4. **Secondary (AOV among converters)** — **Welch's t-test** + **Mann-Whitney**
   (non-parametric robustness).
5. **Combined (revenue per user)** — the metric that ties conversion and AOV
   together and actually drives the decision.
6. **Power & segments** — achieved power, the per-arm sample size needed for 80%
   power, and a device-level cut flagged as directional only.

## Results (synthetic, seed 42)

- **SRM:** balanced (p = 0.14) — randomisation held.
- **Conversion:** 5.17% → 6.00%, **+16.0% relative** (+0.83pp), 95% CI
  [+0.38, +1.28] pp, z = 3.60, **p = 0.0003 → significant**.
- **AOV among converters:** +INR 19, **not significant** (p = 0.32) — correctly
  no false claim.
- **Revenue per user:** +17.5%, **p = 0.0002 → significant**.
- **Power:** 0.95 at the observed effect; ~12,065 per arm needed for 80% power.

**Recommendation:** ship the redesign. The conversion and revenue-per-user gains
are significant and sizeable; the AOV move is noise and shouldn't be claimed.

## Run it

```bash
python generate_data.py     # -> data/experiment.csv
python ab_test.py           # prints the full analysis, writes outputs/
```

## Outputs

| File | What it is |
|------|------------|
| `outputs/results_summary.csv` | Conversion & revenue-per-user: lift, p-value, verdict |
| `outputs/segment_breakdown.csv` | Conversion by device (directional) |
| `outputs/conversion_result.png` | Conversion by variant with 95% CIs |

## Statistical hygiene shown here

Checking SRM before analysis, pre-committing to a primary metric, using CIs not
just p-values, reporting revenue-per-user rather than cherry-picking, and
treating per-segment results as hypotheses (multiple-comparison risk) rather than
decisions.
