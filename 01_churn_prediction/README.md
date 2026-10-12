# 01 — Customer Churn Prediction

Predict which customers are likely to churn so the retention team can target them
before they lapse, and explain *why* each is at risk.

## Problem

For a beauty e-commerce platform, identify customers at high risk of churning
(no purchase in the next 90 days). Churn is costly: winning back a lapsed
customer is far more expensive than retaining an active one, so the business
wants a ranked, explainable at-risk list.

## Approach

1. **Data** — `generate_data.py` creates 20,000 synthetic customers with RFM,
   engagement (app opens, sessions), catalogue breadth, discount dependence,
   service friction and loyalty tier. A churn label is drawn from a latent
   propensity so recency, engagement and habit genuinely drive the outcome.
2. **Model** — a scikit-learn `Pipeline` (scaling + one-hot) trains and compares
   **Logistic Regression** (interpretable, balanced class weights) and
   **Gradient Boosting**, evaluated on a stratified 25% hold-out.
3. **Explain** — permutation importance on the better model surfaces the real
   churn drivers, not just coefficients.
4. **Action** — the whole base is scored; customers are banded Low / Medium /
   High / Critical and the High+Critical list is exported for CRM.

## Results (synthetic, seed 42)

- **ROC-AUC ≈ 0.88**, PR-AUC ≈ 0.83.
- Top drivers: days since last order, recent sessions, order frequency, discount
  dependence — i.e. recency and engagement, as expected.
- Flags the top ~20% of the base as High/Critical risk for intervention.

## Run it

```bash
python generate_data.py     # -> data/customers.csv
python churn_model.py       # -> outputs/
```

## Outputs

| File | What it is |
|------|------------|
| `outputs/model_metrics.csv` | ROC-AUC / PR-AUC per model |
| `outputs/roc_curve.png` | ROC comparison of both models |
| `outputs/confusion_matrix.png` | Hold-out confusion matrix (best model) |
| `outputs/feature_importance.png` | Permutation importance — the churn drivers |
| `outputs/at_risk_customers.csv` | Ranked High/Critical-risk customers to action |

## What this would look like in production

Schedule the scoring to run nightly on the warehouse, write `churn_score` and
`risk_band` back to a customer table, and feed the High/Critical band into the
CRM / campaign tool. Monitor AUC drift and re-train when it degrades.
