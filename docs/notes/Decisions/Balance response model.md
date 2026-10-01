# Balance response to a candidate limit: a documented assumption, not a fitted model

> [!note] Decision
> `balance(L) = min(current_balance × (L / current_limit)^elasticity, L)`, anchored exactly at each customer's own observed `(current_limit, current_balance)` point, `elasticity = 0.3` (`optimization.balance_elasticity` in config). This is the single formula everything downstream — profitability, both optimization stages, stress testing, Monte Carlo's EAD — reuses to ask "what would this customer's balance be at a *different* limit."

> [!tip] Why not derive it from the EAD or PD models instead
> This generator never varies a customer's `credit_limit` over their 36-month history, so `credit_exposure` (the EAD model's stand-in for credit limit) had rank 13 of 46 and importance 0.0048 in the fitted EAD regressor — the model never saw within-customer limit variation to learn a genuine causal relationship from, only between-customer correlation confounded by segment/income. Substituting a different `credit_exposure` into that model's feature vector would extrapolate noise, not a real counterfactual. Same reasoning rules out a candidate-limit-varying PD — it's held constant at each customer's observed value everywhere a candidate limit changes.

> [!warning] Trade-off accepted
> `elasticity = 0.3` is a documented assumption, not fit from data — there is no ground truth to fit it against in this dataset. `elasticity < 1` gives diminishing returns by construction (balance grows slower than the candidate limit), which is the concave Limit-vs-Profit shape a real profitability curve should have, but the exact curvature is a modeling choice, stated as one.

> [!example] Where this shows up in the code
> - `models/profitability.py`'s `balance_at_limit`/`evaluate_at_limit` — the one place the formula is implemented
> - `optimization/customer_optimizer.py` and `optimization/portfolio_optimizer.py` — evaluate every candidate in the €500-€10,000 grid through it
> - `simulation/scenarios.py` — evaluates the FUNDED book's economics at each customer's Phase 5 limit through the same function, so a stress scenario's Expected Loss moves through PD, not an invented balance shock

The anchor-point property (`balance(current_limit) == current_balance` exactly) is regression-tested in `tests/test_profitability.py::test_balance_at_limit_anchor_point_exact` and re-verified against real data on every `make economics` run (`anchor_point_max_abs_diff` in the profitability report).

See also: [[Home]]
