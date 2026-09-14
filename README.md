# Headroom

**Dynamic Credit Limit Optimization for a Digital Bank**

> A risk-adjusted credit decision engine combining machine learning, financial
> economics, and mathematical optimization.

A decision-intelligence platform for a digital bank/neobank: it validates messy
customer/transaction data, models probability of default and expected loss,
estimates customer profitability, and optimizes credit limits — per customer and
across the portfolio — subject to risk and exposure constraints, with Monte Carlo
loss simulation and stress testing on top.

> Status: project scaffolding in progress. This README will be filled in with
> architecture, methodology and actual results as each phase is built — no
> fabricated numbers.

## Roadmap

- [x] Phase 1 — foundation (structure, config, logging)
- [ ] Phase 1 — synthetic data generator, validation, tests
- [ ] Phase 2 — EDA, feature engineering, data quality reporting
- [ ] Phase 3 — risk models (Logistic Regression, XGBoost, calibration, SHAP)
- [ ] Phase 4 — economics (revenue, funding cost, expected loss, profitability)
- [ ] Phase 5 — optimization (individual, portfolio, decision policy)
- [ ] Phase 6 — risk management (scenarios, Monte Carlo, VaR/CVaR, monitoring)
- [ ] Phase 7 — dashboard, documentation, final audit

## Quickstart

```bash
make install
make generate-data
make validate-data
make test
```
