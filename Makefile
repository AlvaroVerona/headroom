.PHONY: install generate-data validate-data quality-report eda features train train-xgboost calibrate explain economics optimize-customer optimize-portfolio simulate stress monitor test dashboard all clean

PYTHON := python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

generate-data:
	$(PYTHON) -m credit_limit_optimizer.data.generator

validate-data:
	$(PYTHON) -m credit_limit_optimizer.data.validation

quality-report:
	$(PYTHON) -m credit_limit_optimizer.data.quality_report

eda:
	$(PYTHON) -m credit_limit_optimizer.analysis.eda

features:
	$(PYTHON) -m credit_limit_optimizer.features.engineering

train:
	$(PYTHON) -m credit_limit_optimizer.models.risk_model

train-xgboost:
	$(PYTHON) -m credit_limit_optimizer.models.xgboost_model

calibrate:
	$(PYTHON) -m credit_limit_optimizer.models.calibration

explain:
	$(PYTHON) -m credit_limit_optimizer.models.explain

economics:
	$(PYTHON) -m credit_limit_optimizer.models.revenue_model
	$(PYTHON) -m credit_limit_optimizer.models.profitability

optimize-customer:
	$(PYTHON) -m credit_limit_optimizer.optimization.customer_optimizer

optimize-portfolio:
	$(PYTHON) -m credit_limit_optimizer.optimization.portfolio_optimizer

simulate:
	$(PYTHON) -m credit_limit_optimizer.simulation.monte_carlo

stress:
	$(PYTHON) -m credit_limit_optimizer.simulation.scenarios

monitor:
	$(PYTHON) -m credit_limit_optimizer.monitoring.drift

test:
	$(PYTHON) -m pytest -v

dashboard:
	streamlit run app/streamlit_app.py

all: generate-data validate-data quality-report eda features train train-xgboost calibrate explain economics optimize-customer optimize-portfolio simulate stress monitor test

clean:
	rm -rf data/raw/*.csv data/raw/*.parquet data/processed/*.csv data/processed/*.parquet data/quarantine/*.csv models/*.pkl models/*.joblib models/*.json reports/figures/* reports/outputs/*
