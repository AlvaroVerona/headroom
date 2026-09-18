# Asset correlation: Basel's own QRRE constant, not fit to this data

> [!note] Decision
> The Monte Carlo default simulation's single systemic-factor asset correlation (`simulation.asset_correlation` in config) is **0.04**, applied flat across every funded customer regardless of segment.

> [!tip] Why
> This is Basel II/III's own flat correlation constant for Qualifying Revolving Retail Exposures — the regulatory category unsecured credit cards fall into — not a number fit to this synthetic dataset or picked to make the loss distribution look a particular way. It's meaningfully lower than mortgages or corporate exposures because retail defaults are driven far more by idiosyncratic factors (job loss, illness, a specific customer's bad month) than by shared systemic ones, which is exactly the credit-card book this project models.

> [!warning] Trade-off accepted
> A single flat correlation means no segment-specific correlation structure — subprime, near_prime and prime customers all share the same 0.04 in the copula, even though a real bank might reasonably expect subprime defaults to be somewhat more systemically correlated (recession-sensitive) than prime's. Documented as a limitation in the README, not hidden: "Monte Carlo uses a single systemic factor, not a multi-factor model with segment-specific correlations."

> [!example] Where this shows up in the code
> - `simulation/monte_carlo.py`'s `simulate_portfolio_losses` — `latent = sqrt(rho) * Z + sqrt(1-rho) * eps`, the standard Vasicek/ASRF single-factor Gaussian copula
> - `config/settings.yaml`'s `simulation.asset_correlation: 0.04`

Sanity-checked, not just asserted: the simulation's mean loss lands within 0.5% of the analytical `PD × LGD × EAD` sum on every scenario (`tests/test_monte_carlo.py::test_mc_mean_close_to_analytical_el_on_real_data`), and VaR 99% comes out at ~2.4x Expected Loss even at this modest correlation — correlated defaults genuinely fatten the tail past what a plain binomial loss count would show, which is the entire reason to simulate rather than just scale the mean.

See also: [[Home]]
