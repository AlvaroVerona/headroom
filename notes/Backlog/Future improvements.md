# Future improvements

Real gaps, deliberately not fixed for Phase 7 — either because the fix is a
genuine scope decision (needs real external data or a materially harder model)
rather than a quick patch, or because the project's own scope was set at "a
portfolio project demonstrating the full chain," not a production system. Each
item below has its rationale in [[Home|CLAUDE.md]] or the README's own
Limitations section; this note tracks *what to decide next*, not what happened.

## Real bureau/transaction data integration

The whole pipeline runs on synthetic data (`data/generator.py`, seed=42) —
realistic in structure and internally consistent, but not real customer
behavior. Swapping in real data would need to re-validate every calibrated
threshold in `config/settings.yaml` (LGD by segment, `portfolio_exposure_limit`,
`asset_correlation`, etc.) — none of them are guaranteed to still be reasonable
against a real portfolio's actual scale.

## Multi-factor Monte Carlo

See [[Decisions/Monte Carlo correlation]] — the current model uses Basel's flat
0.04 QRRE correlation across every customer. A real risk team might want
segment-specific correlations (subprime more systemically correlated than
prime) or a genuine multi-factor model. Not done because there's no clean,
citable regulatory constant for a multi-factor QRRE model the way there is for
the single-factor one — would need either real data to fit it or an explicitly
invented assumption, neither of which fit this project's "never fabricate a
result" rule cleanly.

## Reactive (not static) portfolio policy

The portfolio MIP (`optimization/portfolio_optimizer.py`) solves once, against
a snapshot. A real bank would re-run this periodically as the book ages and
re-optimize which customers stay funded — a genuinely different (and harder)
problem: today's MIP doesn't know about yesterday's funding decisions or the
cost of *changing* a customer's limit, just today's optimal allocation.

## Stress scenarios don't shock `average_balance` or LGD

See [[Findings/High interest rate profit cancellation]]'s sibling reasoning in
`simulation/scenarios.py`'s module docstring — no causal balance-response-to-
macro-shock model exists (same reason [[Decisions/Balance response model]]
isn't derived from the EAD model), and the spec's own scaffolded
`config/settings.yaml` scenarios never included an LGD shock key. Adding either
would need a genuine modeling decision, not just wiring up an existing formula.

## Real-time monitoring with alerting

`monitoring/drift.py` runs on demand (`make monitor`) against the three fixed
snapshot vintages. A real deployment would run PSI on a schedule against
actual new production data and alert on threshold breaches — meaningfully
different from this project's "no true production data exists past month 24"
situation (see [[Findings/Delinquency count vintage artifact]]).

## Infrastructure

A proper database backend instead of CSVs, MLflow model tracking, cloud
deployment — all straightforward, none started, none blocking anything else.

See also: [[Home]]
