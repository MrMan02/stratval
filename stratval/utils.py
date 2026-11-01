from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import stats

##############################################################################################################

ReturnType = Literal["simple", "log"]

##############################################################################################################

@dataclass(frozen=True)
class ReturnSeries:
    """
    Per-period returns of one strategy, plus the metadata the tests need.
    """

    values: np.ndarray
    periods_per_year: int
    return_type: ReturnType = "log"
    risk_free: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        """
        Validates the input data and sets the attributes.
        """
        values = np.asarray(self.values, dtype = float)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if values.size < 2:
            raise ValueError("at least two observations are needed")
        if not np.all(np.isfinite(values)):
            raise ValueError("values contain NaN or inf")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if self.return_type not in ("simple", "log"):
            raise ValueError("return_type must be 'simple' or 'log'")
        if not np.isfinite(self.risk_free):
            raise ValueError("risk_free must be a finite number")

        # frozen = True blocks normal assignment, so we bypass it once here
        object.__setattr__(self, "values", values)

    @property
    def n_obs(self) -> int:
        """
        Number of observations in the return series.
        """
        return self.values.size


##############################################################################################################

def newey_west_lags(n_obs: int) -> int:
    """
    Rule-of-thumb truncation lag floor(4 (T/100)^(2/9)) (Newey and West, 1994).
    """
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))

def long_run_covariance(phi: np.ndarray, lags : int | None = None) -> np.ndarray:
    """
    Newey-West (1987) estimate of the long-run covariance of the rows of phi.

    phi is a T x k (k = 1 or k = 2) matrix whose row t is phi_t. The rows must already have
    sample mean zero (true for GMM moment vectors at the estimate), so no
    demeaning is done here. Returns the k x k matrix
    Gamma_0 + sum_{j=1}^{m} (1 - j/(m+1)) (Gamma_j + Gamma_j').
    """
    phi = np.asarray(phi, dtype = float)
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]
    if phi.ndim != 2:
        raise ValueError("phi must be a 1D or 2D array")
    n_obs = phi.shape[0]
    if lags is None:
        lags = newey_west_lags(n_obs)
    if not 0 <= lags < n_obs:
        raise ValueError("lags must satisfy 0 <= lags < T")
    
    sigma = phi.T @ phi / n_obs
    for j in range(1, lags + 1):
        gamma_j = phi[j:].T @ phi[:-j] / n_obs
        weight = 1.0 - j / (lags + 1)
        sigma += weight * (gamma_j + gamma_j.T)
        
    return sigma
    
##############################################################################################################

def autocorrelations(x : np.ndarray, max_lag : int) -> np.ndarray:
    """
    Sample autocorrelations rho_1, ..., rho_max_lag of x (divisor T at every lag).
    """
    x = np.asarray(x, dtype = float)
    if not 0 <= max_lag < x.size:
        raise ValueError("max_lag must satisfy 0 <= max_lag < T")
    e  = x - x.mean()
    gamma_0 = e @ e
    
    return np.array([e[j:] @ e[:-j] / gamma_0 for j in range(1, max_lag + 1)])

@dataclass(frozen=True)
class LjungBoxResult:
    """
    Ljung-Box test of H0: rho_1 = ... = rho_h = 0.
    """

    stat: float
    p_value: float
    max_lag: int
    autocorrelations: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)
    
def ljung_box(x: np.ndarray, max_lag: int) -> LjungBoxResult:
    """
    Ljung and Box (1978) portmanteau test.

    Q = T (T + 2) sum_{k=1}^{h} rho_k^2 / (T - k), asymptotically chi^2(h) under H0
    that the first h autocorrelations are zero.
    """
    x = np.asarray(x, dtype=float)
    n_obs = x.size
    if not 1 <= max_lag < n_obs:
        raise ValueError("max_lag must satisfy 1 <= max_lag < n_obs")
    rho = autocorrelations(x, max_lag)
    k = np.arange(1, max_lag + 1)
    q_stat = n_obs * (n_obs + 2) * np.sum(rho**2 / (n_obs - k))
    
    return LjungBoxResult(
        stat = float(q_stat),
        p_value = float(stats.chi2.sf(q_stat, df=max_lag)),
        max_lag = max_lag,
        autocorrelations = tuple(float(r) for r in rho),
    )