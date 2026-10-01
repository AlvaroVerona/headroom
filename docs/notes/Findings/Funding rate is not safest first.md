# Prime, the safest segment, gets funded the LEAST

> [!bug] The counter-intuitive result
> At the portfolio-optimization MIP's solution: prime (the lowest-risk segment) has a **20%** funded rate — the lowest of the three segments — while near_prime is funded at **72%**. Subprime, the riskiest segment, sits in between.

> [!tip] Why this is correct, not a bug
> Exposure is the binding portfolio constraint here (funding every approved customer would need €320.5M against a €150M limit), not credit risk — Expected Loss and average PD both have slack at the MIP-optimal solution. The MIP is implicitly maximizing profit *per euro of scarce exposure budget*, and prime customers carry large individually-optimal limits (per [[Decisions/Balance response model]] and the income-multiple constraint) for comparatively modest absolute profit: prime's profit-per-euro-of-exposure is **€0.022**, the segment's worst. Subprime has the BEST profit-per-euro (**€0.057**) but is capped by the ALSO-binding high-risk exposure constraint (max 15% of exposure in subprime), leaving near_prime with the most headroom under both binding constraints simultaneously — hence its highest funded rate.

> [!success] Verified, not just plausible
> Checked directly against the real solved MIP: exposure and high-risk exposure both land exactly at their configured limits (not with slack), confirming they're the constraints actually driving the allocation, and the two ratio-constraint linearizations (average PD, high-risk exposure %) were verified to match the TRUE non-linearized ratio recomputed on the selected set (`tests/test_portfolio_optimizer.py::test_average_pd_linearization_matches_true_ratio`, `test_high_risk_exposure_linearization_matches_true_ratio`).

> [!example] Where this shows up
> `optimization/portfolio_optimizer.py`'s `profit_efficiency_by_segment` and `write_model_card` — the full segment-level profit-per-exposure breakdown is computed and written into `reports/model_cards/portfolio_optimization.md` on every `make optimize-portfolio` run.

See also: [[Home]], [[Decisions/Portfolio vs individual optimization]]
