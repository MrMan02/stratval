from __future__ import annotations
from typing import Literal
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from stratval.utils import ReturnSeries, long_run_covariance, newey_west_lags, autocorrelations

##############################################################################################################

VarianceMethod = Literal["iid_normal", "iid", "hac"]
Alternative = Literal["two-sided", "greater", "less"]
Annualization = Literal["sqrt", "lo"]

##############################################################################################################

def _moments(rs : ReturnSeries) -> tuple[float, float]:
    """
    Sample mean and variance of excess returns, divisor T (Lo, eq. 3).
    """
    
    return float(rs.values.mean()), float(rs.values.var(ddof=0))

def sharpe_ratio(rs : ReturnSeries) -> float:
    """
    Sharpe ratio of excess returns (Lo, eq. 4).
    """
    if np.ptp(rs.values) == 0.0:
        raise ValueError("excess returns are constant: the Sharpe ratio is undefined")
    
    mu, sigma2 = _moments(rs)
    
    return (mu - rs.risk_free) / np.sqrt(sigma2)
     
def _moment_vector(rs : ReturnSeries) -> np.ndarray:
    """
    T x 2 matrix with rows phi_t = (R_t - mu, (R_t - mu)^2 - sigma2) (Lo, A8).
    """
    mu, sigma2 = _moments(rs)
    e = rs.values - mu
    
    return np.column_stack((e, e**2 - sigma2))

def _sharpe_gradient(mu : float, sigma2 : float, risk_free : float) -> np.ndarray:
    """
    Gradient of g(mu, sigma2) = (mu - R_f) / sqrt(sigma2) (Lo, A3).
    """
    sigma = np.sqrt(sigma2)
    
    return np.array([1.0 / sigma, -(mu - risk_free) / (2.0 * sigma**3)])

def sharpe_variance(rs : ReturnSeries, method : VarianceMethod = "hac", lags : int | None = None) -> float:
    """
    Asymptotic variance V of sqrt(T) (SR_hat - SR), so that Var(SR_hat) ~ V / T.

    - "iid_normal": grad' Gamma_0 grad = 1 + SR^2 / 2 (Lo, A4)
    - "iid":        grad' Gamma_0 grad = 1 - g3 SR + (g4 - 1)/4 SR^2 (Mertens, 2002)
    - "hac":        grad' Sigma grad, Sigma by Newey-West with `lags` lags (Lo, A11)
    """
    sr = sharpe_ratio(rs)
    if method == "iid_normal":
        return 1.0 + 0.5 * sr**2
    if method == "iid":
        lags = 0
    elif method != "hac":
        raise ValueError("method must be 'iid_normal', 'iid' or 'hac'")
    
    mu, sigma2 = _moments(rs)
    grad = _sharpe_gradient(mu, sigma2, rs.risk_free)
    sigma_hat = long_run_covariance(_moment_vector(rs), lags = lags)
    
    return float(grad @ sigma_hat @ grad)

@dataclass(frozen=True)
class SharpeTestResult:
    """
    Result of hypothesis test on the Sharpe ratio.(Lo, eq. 5).
    """
    
    sr: float
    sr0: float
    se: float
    stat: float
    p_value: float
    ci: tuple[float, float]
    alpha: float
    alternative: str
    method: str
    lags: int | None
    n_obs: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
def sharpe_test(
    rs: ReturnSeries,
    sr0: float = 0.0,
    method: VarianceMethod = "hac",
    alternative: Alternative = "greater",
    alpha: float = 0.05,
    lags: int | None = None,
) -> SharpeTestResult:
    """
    Asymptotic test of H0: SR vs sr0 and the matching (1 - alpha) confidence interval.

    The interval is two-sided for "two-sided" and one-sided otherwise.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if method == "hac" and lags is None:
        lags = newey_west_lags(rs.n_obs)
    elif method == "iid":
        lags = 0
    elif method == "iid_normal":
        lags = None
        
    sr = float(sharpe_ratio(rs))
    se = float(np.sqrt(sharpe_variance(rs, method = method, lags = lags) / rs.n_obs))
    z = (sr - sr0) / se
    
    if alternative == "two-sided":
        p_value = 2.0 * stats.norm.sf(np.abs(z))
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        ci = (sr - z_crit * se, sr + z_crit * se)
    elif alternative == "greater":
        p_value = stats.norm.sf(z)
        z_crit = stats.norm.ppf(1.0 - alpha)
        ci = (sr - z_crit * se, float("inf"))
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
        z_crit = stats.norm.ppf(alpha)
        ci = (-float("inf"), sr - z_crit * se)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    
    return SharpeTestResult(
        sr = sr,
        sr0 = sr0,
        se = se,
        stat = z,
        p_value = p_value,
        ci = ci,
        alpha = alpha,
        alternative = alternative,
        method = method,
        lags = lags,
        n_obs = rs.n_obs,
    )
    
def eta(rho : npndarray, q : int) -> float:
    """
    Lo's time-aggregation factor, SR(q) = eta(q) SR (Lo, eq. 20).

    rho = (rho_1, ..., rho_K) with K <= q - 1; autocorrelations beyond lag K
    are treated as zero. With rho = 0 it reduces to sqrt(q).
    """
    rho = np.asarray(rho, dtype = float)
    if rho.size > q - 1:
        raise ValueError("at most q - 1 autocorrelations are used")
    k = np.arange(1, rho.size + 1)
    denominator = q + 2.0 * np.sum((q - k) * rho)
    if denominator <= 0.0:
        raise ValueError("autocorrelations imply a non-positive q-period variance")
    
    return float(q / np.sqrt(denominator))

def annualized_sharpe(rs : ReturnSeries, method : Annualization = "lo", max_lag: int | None = None) -> float:
    """
    Annual Sharpe ratio from per-period returns, q = rs.periods_per_year.

    - "sqrt": sqrt(q) SR_hat, valid for IID returns (Lo, eq. 17)
    - "lo":   eta(q) SR_hat with sample autocorrelations up to max_lag (Lo, eq. 20).
              Default max_lag: the Newey-West lag, capped at q - 1.
    Exact for log returns, approximate for simple returns.
    """
    q = rs.periods_per_year
    sr = sharpe_ratio(rs)
    if method == "sqrt":
        return float(np.sqrt(q) * sr)
    if method != "lo":
        raise ValueError("method must be 'sqrt' or 'lo'")
    if max_lag is None:
        max_lag = min(q - 1, newey_west_lags(rs.n_obs))
    rho = autocorrelations(rs.values, max_lag)
    
    return eta(rho, q) * float(sr)

def _aggregation_moment_vector(rs : ReturnSeries, max_lag : int) -> np.ndarray:
    """
    (T - K) x (K + 2) matrix of moment vectors for (mu, sigma2, gamma_1..gamma_K) (Lo, A16).

    Rows start at t = K + 1, where all lags exist; columns are demeaned.
    """
    mu, sigma2 = _moments(rs)
    e = rs.values - mu
    n_obs = rs.n_obs
    k_max = max_lag
    columns = [e[k_max:], e[k_max:] ** 2 - sigma2]
    for k in range(1, k_max + 1):
        gamma_k = e[k:] @ e[:-k] / n_obs
        columns.append(e[k_max:] * e[k_max - k:-k] - gamma_k)
    phi = np.column_stack(columns)
    
    return phi

def _annualized_gradient(mu : float, sigma2 : float, gammas : np.ndarray, q : int, risk_free : float) -> np.ndarray:
    """
    Gradient of g = q (mu - R_f) / sqrt(D), D = sigma2 (q + 2 sum (q - k) gamma_k) (Lo, A23-A25).
    """
    k = np.arange(1, gammas.size + 1)
    d = q * sigma2 + 2.0 * np.sum((q - k) * gammas)
    excess = mu - risk_free
    
    return np.concatenate(
        [
            [q / np.sqrt(d), -(q**2) * excess / (2.0 * d**1.5)],
            -q * (q - k) * excess / d**1.5,
        ]
    )
    
def annualized_sharpe_variance(
    rs : ReturnSeries, 
    method : Annualization = "lo",
    max_lag : int | None = None, 
    variance_method : VarianceMethod = "hac",
    lags : int | None = None
) -> float:
    """
    """
    q = rs.periods_per_year
    if method == "sqrt":
        return q * sharpe_variance(rs, method = variance_method, lags = lags)
    if method != "lo":
        raise ValueError("method must be 'sqrt' or 'lo'")
    if max_lag is None:
        max_lag = min(q - 1, newey_west_lags(rs.n_obs))
    if not 0 <= max_lag < q:
        raise ValueError("max_lag must satisfy 0 <= max_lag <= q - 1!")
    
    mu, sigma2 = _moments(rs)
    gammas = sigma2 * autocorrelations(rs.values, max_lag)
    phi = _aggregation_moment_vector(rs, max_lag)
    sigma_hat = long_run_covariance(phi, lags=lags)
    grad = _annualized_gradient(mu, sigma2, gammas, q, rs.risk_free)
    
    return float(grad @ sigma_hat @ grad)