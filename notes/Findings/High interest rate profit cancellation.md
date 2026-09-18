# `high_interest_rate`'s net profit exactly matches base, to the cent

> [!success] Not a bug — verified
> `high_interest_rate` scenario total Expected Profit: €8,186,016.98. Base scenario: €8,186,016.98. Identical to the cent, confirmed by inspecting both scenarios' revenue and funding-cost breakdowns separately, not just the top-line profit number.

> [!bug] Why this looks like a bug at first
> `high_interest_rate` carries no feature-level shock at all — mean PD is confirmed unchanged from base (`tests/test_scenarios.py::test_high_interest_rate_pd_unchanged_from_base`). Its only two shocks are `funding_rate_shift: +0.03` and `apr_shift: +0.03`, applied to the funding cost and interest revenue formulas respectively. A rate-rise scenario producing *zero* net-profit impact reads like a sign error or a shock not being applied.

> [!tip] What's actually happening
> `interest_revenue = balance × apr` and `funding_cost = balance × funding_rate` both scale the exact same `balance` figure. Two EQUAL-MAGNITUDE shifts (+3pp each) applied to APR and funding rate cancel exactly in net profit — even though gross revenue and gross funding cost both genuinely move by +€1,028,505 each, real and substantial changes that are invisible from the "total profit" number alone.

> [!warning] The real-world risk this illustrates
> If a bank's own repricing (APR) tracks its cost of funds one-for-one, a rate-rise scenario looks like a bottom-line non-event even though NIM composition changed substantially — a genuine planning trap for anyone reading only the headline profit figure instead of the revenue/cost decomposition.

> [!example] Where this shows up
> `simulation/scenarios.py`'s `write_model_card` — this exact comparison is computed and written into `reports/model_cards/scenarios.md` on every `make stress` run, so it's re-verified live each time, not a one-off observation.

See also: [[Home]], [[Decisions/Monte Carlo correlation]]
