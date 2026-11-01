import numpy as np
import pytest

from stratval.sharpe.sharpe import (
    _moment_vector,
    _moments,
    _sharpe_gradient,
    sharpe_ratio,
    sharpe_variance,
    newey_west_lags,
    sharpe_test,
    annualized_sharpe,
    eta,
    _aggregation_moment_vector,
    _annualized_gradient,
    annualized_sharpe_variance,
)
from stratval.utils import ReturnSeries

##############################################################################################################

def test_sharpe_ratio_by_hand():
    # mean 0.02, population std 0.01 -> SR = 2
    rs = ReturnSeries([0.01, 0.03], periods_per_year=12)
    assert sharpe_ratio(rs) == pytest.approx(2.0)


def test_sharpe_ratio_subtracts_risk_free():
    rs = ReturnSeries([0.01, 0.03], periods_per_year=12, risk_free=0.01)
    assert sharpe_ratio(rs) == pytest.approx(1.0)


def test_sharpe_ratio_is_scale_invariant():
    x = np.random.default_rng(0).normal(0.01, 0.05, size=100)
    sr = sharpe_ratio(ReturnSeries(x, periods_per_year=12))
    sr_scaled = sharpe_ratio(ReturnSeries(3 * x, periods_per_year=12))
    assert sr_scaled == pytest.approx(sr)


@pytest.mark.parametrize("n", [3, 100])
def test_constant_returns_raise(n):
    with pytest.raises(ValueError):
        sharpe_ratio(ReturnSeries(np.full(n, 0.01), periods_per_year=12))


def test_moment_vector_has_zero_sample_mean():
    x = np.random.default_rng(1).normal(0.01, 0.05, size=200)
    phi = _moment_vector(ReturnSeries(x, periods_per_year=12))
    assert phi.shape == (200, 2)
    np.testing.assert_allclose(phi.mean(axis=0), 0.0, atol=1e-15)


def test_gradient_matches_finite_differences():
    mu, sigma2, rf = 0.01, 0.0025, 0.002
    g = lambda m, s2: (m - rf) / np.sqrt(s2)
    h = 1e-7
    numerical = np.array([
        (g(mu + h, sigma2) - g(mu - h, sigma2)) / (2 * h),
        (g(mu, sigma2 + h) - g(mu, sigma2 - h)) / (2 * h),
    ])
    np.testing.assert_allclose(_sharpe_gradient(mu, sigma2, rf), numerical, rtol=1e-6)
    
def _series_with_sharpe(sr: float, n_obs: int) -> ReturnSeries:
    # alternating +-1 has mean 0 and population std 1, so SR_hat = sr exactly
    z = np.where(np.arange(n_obs) % 2 == 0, 1.0, -1.0)
    return ReturnSeries(sr + z, periods_per_year=12)


@pytest.mark.parametrize(
    "sr, n_obs, expected_se",
    [(0.50, 12, 0.306), (1.50, 60, 0.188), (3.00, 60, 0.303), (2.00, 500, 0.077)],
)
def test_iid_normal_reproduces_lo_table_1(sr, n_obs, expected_se):
    rs = _series_with_sharpe(sr, n_obs)
    se = np.sqrt(sharpe_variance(rs, method="iid_normal") / n_obs)
    assert se == pytest.approx(expected_se, abs=5e-4)


def test_iid_matches_mertens_formula():
    x = np.random.default_rng(5).standard_t(df=5, size=500) * 0.02 + 0.005
    rs = ReturnSeries(x, periods_per_year=12)
    sr = sharpe_ratio(rs)
    e = x - x.mean()
    g3 = np.mean(e**3) / np.mean(e**2) ** 1.5
    g4 = np.mean(e**4) / np.mean(e**2) ** 2
    mertens = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    assert sharpe_variance(rs, method="iid") == pytest.approx(mertens)


def test_hac_with_zero_lags_equals_iid():
    x = np.random.default_rng(6).normal(0.01, 0.05, size=300)
    rs = ReturnSeries(x, periods_per_year=12)
    assert sharpe_variance(rs, method="hac", lags=0) == pytest.approx(
        sharpe_variance(rs, method="iid")
    )


def test_methods_agree_on_long_iid_normal_sample():
    x = np.random.default_rng(7).normal(0.01, 0.05, size=100_000)
    rs = ReturnSeries(x, periods_per_year=12)
    v_normal = sharpe_variance(rs, method="iid_normal")
    assert sharpe_variance(rs, method="iid") == pytest.approx(v_normal, rel=0.02)
    assert sharpe_variance(rs, method="hac") == pytest.approx(v_normal, rel=0.05)


def test_hac_detects_positive_autocorrelation():
    rho, n = 0.5, 20_000
    eps = np.random.default_rng(8).normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    rs = ReturnSeries(0.1 + x, periods_per_year=12)
    # small SR: V_hac ~ (1 + rho)/(1 - rho) = 3, V_iid ~ 1
    assert sharpe_variance(rs, method="hac", lags=50) > 2.5
    assert sharpe_variance(rs, method="iid") < 1.1


def test_unknown_method_raises():
    rs = ReturnSeries([0.01, 0.03], periods_per_year=12)
    with pytest.raises(ValueError):
        sharpe_variance(rs, method="bootstrap")
        
def test_known_two_sided_p_value():
    # SR = 0.25, T = 60, IID normal: se = sqrt(1.03125/60) = 0.13110, z = 1.9069
    rs = _series_with_sharpe(0.25, 60)
    res = sharpe_test(rs, method="iid_normal", alternative="two-sided")
    assert res.se == pytest.approx(0.13110, abs=1e-5)
    assert res.stat == pytest.approx(1.9069, abs=1e-4)
    assert res.p_value == pytest.approx(0.05653, abs=1e-5)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_ci_is_dual_to_test(alternative):
    # testing sr0 = a finite CI bound gives p-value exactly alpha
    x = np.random.default_rng(9).normal(0.01, 0.05, size=240)
    rs = ReturnSeries(x, periods_per_year=12)
    res = sharpe_test(rs, alternative=alternative, alpha=0.10)
    bound = res.ci[0] if np.isfinite(res.ci[0]) else res.ci[1]
    res_at_bound = sharpe_test(rs, sr0=bound, alternative=alternative, alpha=0.10)
    assert res_at_bound.p_value == pytest.approx(0.10)


def test_one_sided_p_values_add_up():
    x = np.random.default_rng(10).normal(0.01, 0.05, size=240)
    rs = ReturnSeries(x, periods_per_year=12)
    p_greater = sharpe_test(rs, alternative="greater").p_value
    p_less = sharpe_test(rs, alternative="less").p_value
    p_two = sharpe_test(rs, alternative="two-sided").p_value
    assert p_greater + p_less == pytest.approx(1.0)
    assert p_two == pytest.approx(2 * min(p_greater, p_less))


def test_result_records_settings():
    x = np.random.default_rng(11).normal(0.01, 0.05, size=240)
    rs = ReturnSeries(x, periods_per_year=12)
    res = sharpe_test(rs)
    assert res.method == "hac"
    assert res.lags == newey_west_lags(240)
    assert res.n_obs == 240
    assert set(res.to_dict()) >= {"sr", "se", "p_value", "ci", "method", "lags"}


def test_ci_coverage_iid_normal():
    # true SR = 0.2; the 95% two-sided CI should cover it about 95% of the time
    rng = np.random.default_rng(12)
    n_sims, n_obs, true_sr = 2000, 240, 0.2
    covered = 0
    for _ in range(n_sims):
        rs = ReturnSeries(rng.normal(true_sr, 1.0, size=n_obs), periods_per_year=12)
        lo, hi = sharpe_test(rs, method="iid_normal", alternative="two-sided").ci
        covered += lo <= true_sr <= hi
    assert 0.93 <= covered / n_sims <= 0.97


@pytest.mark.parametrize(
    "kwargs", [dict(alpha=0.0), dict(alpha=1.0), dict(alternative="bigger")]
)
def test_invalid_test_arguments_raise(kwargs):
    rs = ReturnSeries([0.01, 0.03, 0.02], periods_per_year=12)
    with pytest.raises(ValueError):
        sharpe_test(rs, **kwargs)
        
def _ar1(rho, n, seed, mean=0.0):
    eps = np.random.default_rng(seed).normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return mean + x


def test_eta_without_autocorrelation_is_sqrt_q():
    assert eta(np.zeros(11), 12) == pytest.approx(np.sqrt(12))
    assert eta(np.array([]), 12) == pytest.approx(np.sqrt(12))


def test_eta_by_hand():
    # q = 2, rho_1 = 0.5: 2 / sqrt(2 + 2 * 1 * 0.5) = 2 / sqrt(3)
    assert eta(np.array([0.5]), 2) == pytest.approx(2 / np.sqrt(3))


def test_eta_matches_ar1_closed_form():
    rho, q = 0.3, 12
    closed_form = q / np.sqrt(q + 2 * rho / (1 - rho) * (q - (1 - rho**q) / (1 - rho)))
    assert eta(rho ** np.arange(1, q), q) == pytest.approx(closed_form)


def test_eta_direction_of_correction():
    assert eta(np.array([0.2]), 12) < np.sqrt(12)   # positive autocorrelation
    assert eta(np.array([-0.2]), 12) > np.sqrt(12)  # negative autocorrelation


@pytest.mark.parametrize("rho, q", [(np.zeros(12), 12), (np.array([-1.0]), 2)])
def test_eta_invalid_inputs_raise(rho, q):
    with pytest.raises(ValueError):
        eta(rho, q)


def test_annualized_sqrt():
    rs = _series_with_sharpe(0.2, 240)
    assert annualized_sharpe(rs, method="sqrt") == pytest.approx(0.2 * np.sqrt(12))


def test_annualized_lo_close_to_sqrt_for_iid():
    x = np.random.default_rng(14).normal(0.01, 0.05, size=100_000)
    rs = ReturnSeries(x, periods_per_year=12)
    assert annualized_sharpe(rs, method="lo") == pytest.approx(
        annualized_sharpe(rs, method="sqrt"), rel=0.02
    )


def test_annualized_lo_corrects_ar1():
    rho, q = 0.4, 12
    rs = ReturnSeries(_ar1(rho, 100_000, seed=15, mean=0.2), periods_per_year=q)
    ratio = annualized_sharpe(rs, method="lo") / annualized_sharpe(rs, method="sqrt")
    expected = eta(rho ** np.arange(1, q), q) / np.sqrt(q)  # about 0.68
    assert ratio == pytest.approx(expected, rel=0.03)


def test_annualized_invalid_arguments_raise():
    rs = _series_with_sharpe(0.2, 240)
    with pytest.raises(ValueError):
        annualized_sharpe(rs, method="lo", max_lag=12)  # more than q - 1 = 11
    with pytest.raises(ValueError):
        annualized_sharpe(rs, method="ar1")
        
@pytest.mark.parametrize("rho, expected", [(0.2, 2.88), (-0.2, 4.17), (0.5, 2.12)])
def test_eta_reproduces_lo_table_2(rho, expected):
    assert eta(rho ** np.arange(1, 12), 12) == pytest.approx(expected, abs=0.005)


def test_annualized_gradient_matches_finite_differences():
    q, rf = 12, 0.002
    theta = np.array([0.01, 0.0025, 0.0004, 0.0001, -0.00005])

    def g(t):
        k = np.arange(1, t.size - 1)
        return q * (t[0] - rf) / np.sqrt(q * t[1] + 2 * np.sum((q - k) * t[2:]))

    h = 1e-8
    numerical = np.array(
        [(g(theta + h * np.eye(5)[i]) - g(theta - h * np.eye(5)[i])) / (2 * h) for i in range(5)]
    )
    analytic = _annualized_gradient(theta[0], theta[1], theta[2:], q, rf)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5)


def test_aggregation_moment_vector_shape():
    x = np.random.default_rng(16).normal(0.01, 0.05, size=300)
    phi = _aggregation_moment_vector(ReturnSeries(x, periods_per_year=12), max_lag=4)
    assert phi.shape == (296, 6)


def test_sqrt_variance_is_q_times_per_period():
    x = np.random.default_rng(17).normal(0.01, 0.05, size=300)
    rs = ReturnSeries(x, periods_per_year=12)
    for vm in ["iid_normal", "iid", "hac"]:
        assert annualized_sharpe_variance(rs, "sqrt", variance_method=vm) == pytest.approx(
            12 * sharpe_variance(rs, method=vm)
        )


def test_lo_variance_with_no_autocovariances_equals_sqrt():
    # K = 0: theta = (mu, sigma2), gradient = sqrt(q) * per-period gradient
    x = np.random.default_rng(18).normal(0.01, 0.05, size=300)
    rs = ReturnSeries(x, periods_per_year=12)
    assert annualized_sharpe_variance(rs, "lo", max_lag=0, lags=3) == pytest.approx(
        annualized_sharpe_variance(rs, "sqrt", variance_method="hac", lags=3)
    )


def test_lo_variance_matches_lo_a32_for_iid():
    q = 12
    x = np.random.default_rng(19).normal(0.01, 0.04, size=200_000)
    rs = ReturnSeries(x, periods_per_year=q)
    sr = sharpe_ratio(rs)
    k = np.arange(1, q)
    a32 = q * (1 + 0.5 * sr**2) + q * sr**2 * np.sum((1 - k / q) ** 2)
    v = annualized_sharpe_variance(rs, "lo", max_lag=q - 1, lags=0)
    assert v == pytest.approx(a32, rel=0.02)


def test_lo_se_matches_simulated_sd_for_ar1():
    # AR(1), rho = 0.3: estimated SE close to the true sampling SD of SR_hat(12)
    rho, n_obs, n_sims = 0.3, 600, 400
    rng = np.random.default_rng(20)
    estimates, ses = [], []
    for _ in range(n_sims):
        u = rng.normal(0.0, np.sqrt(1 - rho**2), size=n_obs)
        e = np.empty(n_obs)
        e[0] = rng.normal()
        for t in range(1, n_obs):
            e[t] = rho * e[t - 1] + u[t]
        rs = ReturnSeries(0.2 + e, periods_per_year=12)
        estimates.append(annualized_sharpe(rs, "lo", max_lag=11))
        ses.append(np.sqrt(annualized_sharpe_variance(rs, "lo", max_lag=11) / n_obs))
    assert np.mean(ses) == pytest.approx(np.std(estimates), rel=0.15)


def test_annualized_variance_invalid_arguments_raise():
    rs = _series_with_sharpe(0.2, 240)
    with pytest.raises(ValueError):
        annualized_sharpe_variance(rs, "lo", max_lag=12)
    with pytest.raises(ValueError):
        annualized_sharpe_variance(rs, "ar1")