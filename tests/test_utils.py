import dataclasses

import numpy as np
import pytest
from scipy import stats

from stratval.utils import (
    ReturnSeries,
    long_run_covariance,
    newey_west_lags,
    autocorrelations,
    ljung_box
)

##############################################################################################################

def test_accepts_list_and_converts_to_float_array():
    rs = ReturnSeries([0.01, -0.02, 0.03], periods_per_year=12)
    assert isinstance(rs.values, np.ndarray)
    assert rs.values.dtype == np.float64
    assert rs.n_obs == 3


def test_default_metadata():
    rs = ReturnSeries([0.01, 0.02], periods_per_year=252)
    assert rs.return_type == "log"
    assert rs.name is None
    assert rs.risk_free == 0.0  # risk_free = 0


def test_is_immutable():
    rs = ReturnSeries([0.01, 0.02], periods_per_year=12)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rs.periods_per_year = 52


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(values=[[0.01, 0.02], [0.03, 0.04]], periods_per_year=12),  # 2-D
        dict(values=[0.01], periods_per_year=12),                        # too short
        dict(values=[0.01, np.nan], periods_per_year=12),                # NaN
        dict(values=[0.01, np.inf], periods_per_year=12),                # inf
        dict(values=[0.01, 0.02], periods_per_year=0),                   # bad frequency
        dict(values=[0.01, 0.02], periods_per_year=12, return_type="pct"),
        dict(values=[0.01, 0.02], periods_per_year=12, risk_free=np.nan),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        ReturnSeries(**kwargs)
        
def test_newey_west_lags_rule():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(500) == 5

def test_zero_lags_is_plain_covariance():
    phi = np.random.default_rng(2).normal(size=(50, 2))
    np.testing.assert_allclose(long_run_covariance(phi, lags=0), phi.T @ phi / 50)


def test_long_run_covariance_by_hand():
    # Gamma_0 = 10/4 = 2.5, Gamma_1 = (-1 - 2 - 4)/4 = -1.75, weight 1/2
    # Sigma = 2.5 + 0.5 * 2 * (-1.75) = 0.75
    phi = np.array([1.0, -1.0, 2.0, -2.0])
    np.testing.assert_allclose(long_run_covariance(phi, lags=1), [[0.75]])


def test_long_run_covariance_is_symmetric_psd():
    phi = np.random.default_rng(3).normal(size=(300, 3))
    phi -= phi.mean(axis=0)
    sigma = long_run_covariance(phi, lags=10)
    np.testing.assert_allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() >= -1e-12


def test_long_run_covariance_of_ar1():
    # AR(1) with rho = 0.5 and unit innovations: long-run variance 1/(1-rho)^2 = 4
    rho, n = 0.5, 1_000_000
    eps = np.random.default_rng(4).normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    x -= x.mean()
    assert long_run_covariance(x, lags=100)[0, 0] == pytest.approx(4.0, rel=0.05)


@pytest.mark.parametrize("lags", [-1, 4])
def test_invalid_lags_raise(lags):
    with pytest.raises(ValueError):
        long_run_covariance(np.ones(4), lags=lags)
        
def test_autocorrelations_of_ar1():
    rho, n = 0.4, 200_000
    eps = np.random.default_rng(13).normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    np.testing.assert_allclose(autocorrelations(x, 3), [0.4, 0.16, 0.064], atol=0.01)


def test_autocorrelations_zero_lags_is_empty():
    assert autocorrelations(np.arange(5.0), 0).size == 0
    
def test_ljung_box_matches_statsmodels_value():
    # reference value from statsmodels.stats.diagnostic.acorr_ljungbox(x, lags=[6])
    x = np.random.default_rng(21).normal(size=120)
    res = ljung_box(x, max_lag=6)
    assert res.stat == pytest.approx(2.835748, abs=1e-6)
    assert res.p_value == pytest.approx(0.829162, abs=1e-6)
    assert len(res.autocorrelations) == 6


def test_ljung_box_size_under_iid():
    rng = np.random.default_rng(22)
    rejections = sum(ljung_box(rng.normal(size=240), 10).p_value < 0.05 for _ in range(2000))
    assert 0.03 <= rejections / 2000 <= 0.07


def test_ljung_box_detects_ar1():
    rho, n = 0.3, 240
    eps = np.random.default_rng(23).normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    assert ljung_box(x, 10).p_value < 0.01


@pytest.mark.parametrize("max_lag", [0, 5])
def test_ljung_box_invalid_lag_raises(max_lag):
    with pytest.raises(ValueError):
        ljung_box(np.arange(5.0), max_lag)