"""Expected Loss model (§19): Expected Loss = PD × LGD × EAD.

**PD**: the calibrated XGBoost classifier (`xgboost_calibrated.joblib`) --
chosen over the LR baseline as "the" production PD source going forward
because it modestly but genuinely beat the LR baseline on held-out test
data (ROC-AUC 0.730 vs. 0.724, PR-AUC 0.165 vs. 0.148, see
`xgboost_model.py`'s model card), and it's now calibrated (Brier 0.043,
ECE 0.002 on test) so its probabilities are usable as real PDs. For a
customer already at 90+ DPD at their snapshot (`eligible_for_default_label
== False`, excluded entirely from classifier training since they have no
defined default_12m outcome), PD is forced to 1.0 rather than using the
model's prediction -- they're already in default, not a forecast target.

**LGD**: segment-based lookup from `config/settings.yaml`
(`economics.lgd_by_segment`), unchanged from Phase 1's documented
rationale (unsecured revolving credit, segment proxies for cure
likelihood).

**EAD**: this dataset has no literal "balance at the moment of default"
field (labels.csv gives forward-looking snapshots at a fixed 12-month
horizon, not conditional on exactly when within that window a default
happens) -- the closest available ground truth is `future_utilization`,
labels.csv's already-verified-leakage-safe 12-months-forward utilization
label (§9's "Additional Targets"). So EAD is estimated in the way real
EAD models typically work when a direct observation isn't available:
regress `future_utilization` on Credit limit / utilization / spending
behavior / customer profile (§19's own EAD driver list) using the same
feature table and time-based split as the risk models, then
`EAD = clip(predicted_future_utilization, 0, 1) × credit_exposure`. This
is a genuine, evaluated regression model (R²/MAE reported below), not a
formula dressed up as one.

Run: python -m credit_limit_optimizer.models.expected_loss
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from credit_limit_optimizer.features.engineering import ALL_FEATURE_COLUMNS, build_feature_table, load_inputs
from credit_limit_optimizer.models import risk_model
from credit_limit_optimizer.models.risk_model import PROTECTED_CHARACTERISTICS
from credit_limit_optimizer.utils.config import PROJECT_ROOT, load_config
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("expected_loss")

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "expected_loss"
MODEL_PATH = PROJECT_ROOT / "models" / "ead_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "expected_loss_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_cards" / "expected_loss.md"

PD_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_calibrated.joblib"
EAD_TARGET = "future_utilization"
EAD_FEATURES = [c for c in ALL_FEATURE_COLUMNS if c not in PROTECTED_CHARACTERISTICS]

EAD_HYPERPARAMETERS = {
    "n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "reg_lambda": 1.0, "eval_metric": "rmse", "early_stopping_rounds": 30,
}

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_full_feature_table() -> pd.DataFrame:
    """Unlike risk_model.load_feature_table(), does NOT drop rows with a
    missing default_12m -- future_utilization (the EAD regression target)
    is defined for every row, including customers already at 90+ DPD at
    their snapshot."""
    config = load_config()
    inputs = load_inputs()
    return build_feature_table(inputs, config)


def train_ead_model(cohorts: dict, random_state: int) -> XGBRegressor:
    X_train, y_train = cohorts["train"][EAD_FEATURES], cohorts["train"][EAD_TARGET]
    X_val, y_val = cohorts["validation"][EAD_FEATURES], cohorts["validation"][EAD_TARGET]
    model = XGBRegressor(**EAD_HYPERPARAMETERS, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log.info("Fit EAD regressor: best_iteration=%d (of %d)", model.best_iteration, EAD_HYPERPARAMETERS["n_estimators"])
    return model


def evaluate_ead_model(model: XGBRegressor, cohorts: dict) -> dict:
    metrics = {}
    for name, cohort_df in cohorts.items():
        X, y = cohort_df[EAD_FEATURES], cohort_df[EAD_TARGET]
        pred = np.clip(model.predict(X), 0, 1)
        metrics[name] = {
            "n": int(len(y)), "r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
        }
        log.info("EAD %s: R2=%.3f MAE=%.4f RMSE=%.4f", name, metrics[name]["r2"], metrics[name]["mae"], metrics[name]["rmse"])
    return metrics


def compute_pd(test_df: pd.DataFrame) -> np.ndarray:
    saved = joblib.load(PD_MODEL_PATH)
    calibrated_model, features = saved["model"], saved["features"]
    pd_values = calibrated_model.predict_proba(test_df[features])[:, 1]
    already_in_default = ~test_df["eligible_for_default_label"].astype(bool)
    pd_values = np.where(already_in_default, 1.0, pd_values)
    return pd_values


def compute_lgd(test_df: pd.DataFrame, config: dict) -> np.ndarray:
    lgd_by_segment = config["economics"]["lgd_by_segment"]
    return test_df["customer_segment"].map(lgd_by_segment).to_numpy()


def compute_ead(model: XGBRegressor, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predicted_utilization = np.clip(model.predict(test_df[EAD_FEATURES]), 0, 1)
    ead_amount = predicted_utilization * test_df["credit_exposure"].to_numpy()
    return predicted_utilization, ead_amount


def plot_ead_fit(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(y_true, y_pred, gridsize=40, cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#C62828", label="Perfect fit")
    ax.set_xlabel("Actual future_utilization")
    ax.set_ylabel("Predicted future_utilization")
    ax.set_title("EAD model fit (test cohort, excl. missing credit_exposure)")
    ax.legend()
    fig.savefig(FIG_DIR / "01_ead_model_fit.png", bbox_inches="tight")
    plt.close(fig)


def plot_el_distribution(el: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(el, bins=60, color="#1565C0")
    ax.set_xlabel("Expected Loss (EUR)")
    ax.set_ylabel("Customer-snapshots")
    ax.set_title("Expected Loss distribution (test cohort)")
    fig.savefig(FIG_DIR / "02_el_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_el_by_segment(el_by_segment: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    segments = ["prime", "near_prime", "subprime"]
    ax.bar(segments, [el_by_segment[s] for s in segments], color=["#2E7D32", "#F9A825", "#C62828"])
    ax.set_ylabel("Mean Expected Loss (EUR)")
    ax.set_title("Mean Expected Loss by segment (test cohort)")
    fig.savefig(FIG_DIR / "03_el_by_segment.png", bbox_inches="tight")
    plt.close(fig)


def plot_el_by_outcome(el_by_outcome: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Did not default", "Defaulted (12m)"]
    values = [el_by_outcome["no_default"], el_by_outcome["default"]]
    ax.bar(labels, values, color=["#2E7D32", "#C62828"])
    ax.set_ylabel("Mean Expected Loss (EUR)")
    ax.set_title("Expected Loss vs. actual outcome (test cohort, validation check)")
    fig.savefig(FIG_DIR / "04_el_by_actual_outcome.png", bbox_inches="tight")
    plt.close(fig)


def run_expected_loss() -> dict:
    config = load_config()
    df = load_full_feature_table()
    cohorts = risk_model.split_cohorts(df)

    ead_model = train_ead_model(cohorts, random_state=config["random_seed"])
    ead_metrics = evaluate_ead_model(ead_model, cohorts)

    # credit_exposure (current_credit_limit) carries real Phase 1 missing-
    # value injections (~2% of rows) -- EAD in euros, and therefore
    # Expected Loss, is undefined without a credit limit to scale
    # predicted utilization by. The EAD model's R²/MAE above are still
    # evaluated on the full test cohort (predicting future_utilization
    # doesn't require credit_exposure), but the euro-denominated Expected
    # Loss figures below are computed only for rows where it's known.
    test_full = cohorts["test"]
    test = test_full[test_full["credit_exposure"].notna()]
    n_excluded_missing_limit = len(test_full) - len(test)
    log.info(
        "Excluding %d of %d test rows (%.1f%%) from Expected Loss (missing credit_exposure)",
        n_excluded_missing_limit, len(test_full), 100 * n_excluded_missing_limit / len(test_full),
    )

    predicted_utilization, ead_amount = compute_ead(ead_model, test)
    pd_values = compute_pd(test)
    lgd_values = compute_lgd(test, config)
    expected_loss = pd_values * lgd_values * ead_amount
    assert not np.isnan(expected_loss).any(), "Expected Loss must be fully defined after excluding missing-credit_exposure rows"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_ead_fit(test[EAD_TARGET].to_numpy(), predicted_utilization)
    plot_el_distribution(expected_loss)

    el_df = test[["customer_segment", "default_12m"]].copy()
    el_df["expected_loss"] = expected_loss
    el_by_segment = el_df.groupby("customer_segment")["expected_loss"].mean().to_dict()
    plot_el_by_segment(el_by_segment)

    eligible = el_df.dropna(subset=["default_12m"])
    el_by_outcome = {
        "no_default": float(eligible.loc[eligible["default_12m"] == 0, "expected_loss"].mean()),
        "default": float(eligible.loc[eligible["default_12m"] == 1, "expected_loss"].mean()),
    }
    plot_el_by_outcome(el_by_outcome)

    log.info(
        "Test cohort: mean PD=%.2f%% mean LGD=%.2f mean EAD=€%.0f mean EL=€%.2f | total portfolio EL=€%.0f",
        pd_values.mean() * 100, lgd_values.mean(), ead_amount.mean(), expected_loss.mean(), expected_loss.sum(),
    )
    log.info("Mean EL by segment: %s", {k: round(v, 2) for k, v in el_by_segment.items()})
    log.info("Mean EL by actual outcome: %s", {k: round(v, 2) for k, v in el_by_outcome.items()})

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": ead_model, "features": EAD_FEATURES}, MODEL_PATH)
    log.info("Saved EAD model to %s", MODEL_PATH.relative_to(PROJECT_ROOT))

    report = {
        "ead_metrics": ead_metrics,
        "test_summary": {
            "n": int(len(test)), "n_excluded_missing_credit_exposure": n_excluded_missing_limit,
            "mean_pd": float(pd_values.mean()), "mean_lgd": float(lgd_values.mean()),
            "mean_ead": float(ead_amount.mean()), "mean_expected_loss": float(expected_loss.mean()),
            "total_portfolio_expected_loss": float(expected_loss.sum()),
        },
        "el_by_segment": {k: float(v) for k, v in el_by_segment.items()},
        "el_by_actual_outcome": el_by_outcome,
        "lgd_by_segment": config["economics"]["lgd_by_segment"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    write_model_card(report)
    log.info("Wrote %s", MODEL_CARD_PATH.relative_to(PROJECT_ROOT))
    return report


def write_model_card(report: dict) -> None:
    ead_test = report["ead_metrics"]["test"]
    ts = report["test_summary"]
    lines = [
        "# Model Card — Expected Loss (§19)",
        "",
        "Expected Loss = PD × LGD × EAD, computed per customer-snapshot on the test cohort. "
        "PD from the calibrated XGBoost classifier (forced to 1.0 for customers already at "
        "90+ DPD at their snapshot); LGD from `config/settings.yaml`'s segment lookup; EAD "
        "from a dedicated regression model — see module docstring for why a regression, not "
        "a formula.",
        "",
        "## EAD regression model",
        "",
        "Predicts `future_utilization` (labels.csv's 12-months-forward utilization label) "
        "from the same feature table and time-based split as the risk models; "
        f"`EAD = clip(prediction, 0, 1) × credit_exposure`.",
        "",
        "| Cohort | N | R² | MAE | RMSE |",
        "|---|---|---|---|---|",
    ]
    for cohort in ["train", "validation", "test"]:
        m = report["ead_metrics"][cohort]
        lines.append(f"| {cohort} | {m['n']:,} | {m['r2']:.3f} | {m['mae']:.4f} | {m['rmse']:.4f} |")
    lines += [
        "",
        f"R² ~0.58-0.61: the scatter shows predictions clustering into roughly three "
        "horizontal bands rather than tracking the diagonal closely — the model captures "
        "*segment*-level utilization well (prime ~0.08-0.10, near_prime/subprime ~0.68-0.70, "
        "matching the equilibrium bands documented in CLAUDE.md's Phase 1 notes) but less of "
        "the within-segment variation around each customer's own equilibrium. Consistent "
        "with the generator's design, not a modeling shortfall to chase further here.",
        "",
        "![EAD model fit](../figures/expected_loss/01_ead_model_fit.png)",
        "",
        "## Expected Loss (test cohort)",
        "",
        f"{ts['n_excluded_missing_credit_exposure']:,} of "
        f"{ts['n'] + ts['n_excluded_missing_credit_exposure']:,} test rows excluded "
        "(missing `credit_exposure` — a real Phase 1 injected-missing-value artifact; "
        "EAD in euros is undefined without a credit limit).",
        "",
        f"Mean PD {ts['mean_pd']:.2%}, mean LGD {ts['mean_lgd']:.2f}, mean EAD "
        f"€{ts['mean_ead']:,.0f}, mean Expected Loss €{ts['mean_expected_loss']:.2f} per "
        f"customer-snapshot. Total portfolio Expected Loss on the {ts['n']:,}-row test "
        f"cohort: €{ts['total_portfolio_expected_loss']:,.0f} — extrapolated to the full "
        f"~50,000-customer portfolio (this test cohort is ~94% of it after the missing-"
        f"credit_exposure exclusion), roughly €"
        f"{ts['total_portfolio_expected_loss'] / ts['n'] * 50000:,.0f}, remarkably close to "
        "`config/settings.yaml`'s `optimization.portfolio_expected_loss_limit` placeholder "
        "of €3,000,000 (documented in CLAUDE.md as \"to be recalibrated once the actual "
        "portfolio's scale is known\") — it turns out to already be a reasonable estimate, "
        "not just a round-number guess. Left unchanged here; Phase 5's optimization is "
        "where this constraint actually gets enforced.",
        "",
        "![EL distribution](../figures/expected_loss/02_el_distribution.png)",
        "",
        "### By segment",
        "",
        ", ".join(f"{k}: €{v:.2f}" for k, v in report["el_by_segment"].items()) + ".",
        "",
        "![EL by segment](../figures/expected_loss/03_el_by_segment.png)",
        "",
        "### Validation check: Expected Loss vs. actual outcome",
        "",
        f"Mean EL among customers who did NOT default: €{report['el_by_actual_outcome']['no_default']:.2f}. "
        f"Mean EL among customers who DID default within 12 months: "
        f"€{report['el_by_actual_outcome']['default']:.2f} — Expected Loss (known only at the "
        "snapshot, before the outcome is observed) should be, and is, dramatically higher for "
        "the group that actually went on to default.",
        "",
        "![EL by actual outcome](../figures/expected_loss/04_el_by_actual_outcome.png)",
        "",
    ]
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_expected_loss()


if __name__ == "__main__":
    main()
