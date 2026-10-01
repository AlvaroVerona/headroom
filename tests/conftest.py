"""Opt-in skipping of tests that need pipeline-generated data.

Many tests are integration tests over the real generated dataset (`make generate-data`
and downstream stages). On a fresh checkout that data doesn't exist. With
HEADROOM_SKIP_MISSING_DATA=1 (set by the fast CI job), a test that fails because a
generated data file is missing is reported as skipped instead of failed. Locally and in
the full-pipeline workflow the variable is unset, so a missing file stays a hard failure.
"""
import os

import pytest

_SKIP = os.environ.get("HEADROOM_SKIP_MISSING_DATA") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not _SKIP or report.passed or call.excinfo is None:
        return
    exc = call.excinfo.value
    if isinstance(exc, FileNotFoundError) and "data/" in str(exc):
        report.outcome = "skipped"
        report.longrepr = (str(item.fspath), item.location[1], "skipped: pipeline data not generated")
