# stratval

**Statistical validation of quantitative trading strategies.**

`stratval` takes the results of a backtest that has already been run, assumed free of look-ahead and survivorship bias, and asks whether those results are statistically credible. It is **not** a backtester: it never generates signals or simulates trades. Its inputs are:

- the return series of the selected strategy, with metadata (sampling frequency, return type, risk-free rate);
- optionally, a benchmark return series;
- optionally, the returns of all the configurations tried during optimization (the "trial matrix"), or walk-forward out-of-sample returns.

It is designed to be generic across asset classes and frequencies, and it validates one strategy at a time.

The tests are grouped into **families**. Each family answers one question about the strategy and lives in its own subpackage. Section *n* of this README describes family *n*: for each test, the idea, the formula and the literature it comes from.

| # | Family | Question | Subpackage | Status |
|---|---|---|---|---|
| 1 | Sharpe ratio inference | Is the Sharpe ratio significantly different from a threshold, and what is it on an annual basis? | `stratval.sharpe` | in progress |

## Installation (Windows)

Requires Python ≥ 3.10.

```powershell
git clone https://github.com/MrMan02/stratval.git
cd stratval
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(On macOS/Linux: `python3 -m venv .venv` and `source .venv/bin/activate`.)

Run the tests with `pytest`.

## Package layout

```
stratval/
├── utils.py            # helpers shared by all families
└── sharpe/
    └── sharpe.py       # family 1: Sharpe ratio inference
tests/
└── test_sharpe.py
```

---

## 1. Sharpe ratio inference

The Sharpe ratio computed on a backtest is an **estimate**, so it has a sampling distribution. This family gives its standard error under increasingly general assumptions on returns, tests $H_0: SR = SR_0$, and annualizes it correctly when returns are serially correlated. The main reference is Lo (2002).

### Notation

| Symbol | Meaning |
|---|---|
| $R_t$, $t = 1, \dots, T$ | per-period returns of the strategy |
| $R_f$ | per-period risk-free rate |
| $\mu = E[R_t]$, $\sigma^2 = \textnormal{Var}(R_t)$ | mean and variance of returns |
| $SR = (\mu - R_f)/\sigma$ | true per-period Sharpe ratio |
| $\hat\mu$, $\hat\sigma^2$ | sample mean and sample variance (divisor $T$) |
| $\widehat{SR} = (\hat\mu - R_f)/\hat\sigma$ | estimated Sharpe ratio |
| $\gamma_3 = E[(R_t - \mu)^3]/\sigma^3$, $\gamma_4 = E[(R_t - \mu)^4]/\sigma^4$ | skewness and kurtosis |
| $\rho_k = \textnormal{Corr}(R_t, R_{t-k})$ | autocorrelation of returns at lag $k$ |
| $q$ | number of periods per year (12 for monthly, 252 for daily, ...) |

All results below are asymptotic: as $T \to \infty$,

$$\sqrt{T}\,\big(\widehat{SR} - SR\big) \xrightarrow{d} N(0, V),$$

and the tests differ only in the assumptions on returns and therefore in $V$. In practice, $\widehat{SR} \approx N(SR, \hat V/T)$ and $SE(\widehat{SR}) = \sqrt{\hat V/T}$.

### 1.1 Standard error under IID normal returns

**Assumption.** Returns are IID and normally distributed.

**Result.** By the delta method applied to $g(\mu, \sigma^2) = (\mu - R_f)/\sigma$ (Lo, 2002, eq. 9 and A4):

$$V_{\text{IID-N}} = 1 + \tfrac{1}{2} SR^2 .$$

The "1" comes from estimating the mean, the "$SR^2/2$" from estimating the volatility. The formula is a benchmark, but normality is rarely plausible for strategy returns, so it tends to understate the uncertainty.

### 1.2 Standard error under IID returns (non-normal)

**Assumption.** Returns are IID with finite fourth moment, not necessarily normal.

**Result.** Keeping the general covariance of $(\hat\mu, \hat\sigma^2)$ in the delta method (Mertens, 2002):

$$V_{\text{IID}} = 1 - \gamma_3\, SR + \frac{\gamma_4 - 1}{4}\, SR^2 .$$

Negative skewness and fat tails, typical of short-volatility strategies, increase the standard error. It reduces to 1.1 when $\gamma_3 = 0$ and $\gamma_4 = 3$. This variance is also the building block of the Probabilistic Sharpe Ratio (Bailey and López de Prado, 2012).

### 1.3 Standard error under stationary, serially dependent returns (HAC)

**Assumption.** Returns are stationary and ergodic, with finite fourth moment and dependence that dies out fast enough for a central limit theorem (Lo, 2002, Appendix A, H1–H6). This allows serial correlation, volatility clustering and non-normality.

**Result.** Write $(\mu, \sigma^2)$ as the solution of the GMM moment conditions $E[\varphi_t] = 0$, with $\varphi_t = \big(R_t - \mu,\ (R_t - \mu)^2 - \sigma^2\big)'$ (Hansen, 1982). Then

$$V_{\text{HAC}} = \nabla g'\, \Sigma\, \nabla g, \qquad \nabla g = \Big(\frac{1}{\sigma},\ -\frac{SR}{2\sigma^2}\Big)', \qquad \Sigma = \sum_{j=-\infty}^{\infty} E[\varphi_t\, \varphi_{t-j}'] ,$$

where $\Sigma$ is the long-run covariance matrix of $\varphi_t$. It is estimated with the Newey and West (1987) estimator:

$$\hat\Sigma = \hat\Gamma_0 + \sum_{j=1}^{m} \Big(1 - \frac{j}{m+1}\Big)\big(\hat\Gamma_j + \hat\Gamma_j'\big), \qquad \hat\Gamma_j = \frac{1}{T} \sum_{t=j+1}^{T} \hat\varphi_t\, \hat\varphi_{t-j}' ,$$

with truncation lag $m \to \infty$, $m/T \to 0$. The default is $m = \lfloor 4\,(T/100)^{2/9} \rfloor$; see Andrews (1991) for data-driven choices. With $m = 0$ the estimator gives back 1.2. This is the most robust of the three and the library's default.

### 1.4 Hypothesis test and confidence interval

Given any of the variances above, the test of $H_0: SR = SR_0$ uses

$$z = \frac{\widehat{SR} - SR_0}{\sqrt{\hat V / T}} \ \overset{a}{\sim}\ N(0, 1),$$

with a one-sided ($SR > SR_0$) or two-sided alternative. The $(1 - \alpha)$ confidence interval is $\widehat{SR} \pm z_{1-\alpha/2}\sqrt{\hat V/T}$. These are large-sample approximations, unreliable with only a few years of monthly data.

### 1.5 Annualization

**IID returns.** The $q$-period return has mean $q\mu$ and variance $q\sigma^2$, so $SR(q) = \sqrt{q}\, SR$ and $V(q) = q\, V$ (Lo, 2002, eq. 17–18).

**Serially correlated returns.** The $q$-period variance includes the autocovariances, and (Lo, 2002, eq. 20)

$$SR(q) = \eta(q)\, SR, \qquad \eta(q) = \frac{q}{\sqrt{q + 2 \sum_{k=1}^{q-1} (q - k)\, \rho_k}} .$$

Positive autocorrelation makes $\sqrt q$ **overstate** the annual Sharpe ratio; Lo reports overstatements above 65% for some hedge funds. Negative autocorrelation makes it understate. The formula is exact for log returns and approximate for simple returns.

Estimating all $q - 1$ autocorrelations is noisy (hopeless for daily data, $q = 252$), so the library offers three estimators, reported side by side:

- **`sqrt`**: $\sqrt q\, \widehat{SR}$.
- **`lo`**: $\eta(q)$ with sample autocorrelations up to lag $K$ and $\rho_k = 0$ beyond. Its standard error follows Lo's GMM derivation (Appendix A, "Time Aggregation").
- **`lo_ar1`**: assume AR(1) returns, $\rho_k = \rho^k$, which gives the closed form
  $$\eta(q) = \frac{q}{\sqrt{q + \dfrac{2\rho}{1-\rho}\Big(q - \dfrac{1 - \rho^q}{1 - \rho}\Big)}} ,$$
  with a single estimated parameter.

Sample autocorrelations and the Ljung and Box (1978) statistic are reported as **diagnostics**, not used to choose the formula: choosing by pre-test would make the reported p-values invalid.

### References

- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817–858.
- Bailey, D. H., and López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3–44.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029–1054.
- Ljung, G. M., and Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297–303.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper, University of Basel.
- Newey, W. K., and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
