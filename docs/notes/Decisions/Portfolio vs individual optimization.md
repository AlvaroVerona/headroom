# Two-stage optimization, not a joint MIP

> [!note] Decision
> Stage 1 (`optimization/customer_optimizer.py`) picks each customer's individually-optimal limit — one independent per-customer argmax over the candidate grid, under income/risk/utilization/DSR constraints. Stage 2 (`optimization/portfolio_optimizer.py`) is a portfolio-level 0/1 knapsack (OR-Tools CBC MIP): one binary variable per already-approved customer, deciding which to actually FUND so the aggregate exposure/loss/risk-concentration budget holds while maximizing total profit. Not a single joint (customer × candidate) MIP.

> [!tip] Why
> Mirrors how real risk-based portfolio triage actually works — score individually first, then allocate scarce risk budget across the approved population. It also keeps the problem tractable: ~38,900 binary variables (one per approved customer) instead of ~500,000+ (one per customer × candidate limit), which a joint formulation would need.

> [!warning] Trade-off accepted
> The two-stage design loses the ability to shrink a customer's limit to help them fit inside the portfolio budget — stage 2 can only fund a customer at their stage-1-optimal limit, or not at all. A joint MIP could pick a smaller, still-profitable limit for a customer who doesn't fit at their individual optimum; this design deliberately doesn't support that, in exchange for tractability.

> [!example] Where this shows up in the code
> - `optimization/customer_optimizer.py`'s `optimize_individual` — stage 1, no OR-Tools involved, a plain per-customer argmax
> - `optimization/portfolio_optimizer.py`'s `build_candidate_pool` (assembles stage 1's output into the pool) and `solve_portfolio` (the actual MIP)
> - `simulation/scenarios.py`'s `build_funded_book` — reuses `build_candidate_pool` + `solve_portfolio` to get the real funded book stress-tested downstream, so Phase 5's decision is the one thing Phase 6 stress-tests, not a re-derived approximation of it

Real result this produced: funding every individually-approved customer would need €320.5M exposure against a €150M limit — a genuinely binding problem. MIP-optimal funds 17,205 of 38,879 (44.3%) for €8.19M/year, 7.77% better than a greedy profit-per-exposure heuristic. See [[Findings/Funding rate is not safest first]] for the non-obvious segment-level result this produced.

See also: [[Home]]
