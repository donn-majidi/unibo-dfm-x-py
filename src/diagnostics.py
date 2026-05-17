import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def adf_wrapper(data: pd.Series, **kwargs):
    """
    This is just a wrapper around the adfuller function.

    Parameters
    ----------
    data : pd.Series
        Input series.

    Raises
    ------
    ValueError
        If the input series contains NaN values.

    Returns
    -------
    out : pd.Series
        ADFULLER test results.

    """
    
    if data.isna().any():
        raise ValueError(f'The input series contains NaN values at index: {data[data.isna()].index}')
    
    results = adfuller(data, **kwargs)
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(results[0:4],index=labels)
    for key, val in results[4].items():
        out[f'critical value ({key})'] = val
    
    return out

def kpss_wrapper(data: pd.Series, **kwargs):
    """
    This is just a wrapper around the kpss function.

    Parameters
    ----------
    data : pd.Series
        Input series.
    **kwargs : Arguments to pass to the KPSS function
        DESCRIPTION.

    Raises
    ------
    ValueError
        If the input series contains NaN values.

    Returns
    -------
    out : pd.Series
        KPSS test results.

    """
    
    if data.isna().any():
        raise ValueError('Ther input series contains NaN values at index: {data[data.isna()].index}')
        
    results = kpss(data, **kwargs)
    labels = ['KPSS test statistic', 'p-value bound', '# lags']
    
    out = pd.Series(results[0:3], index=labels)
    for key, val in results[3].items():
        out[f'critical value ({key})'] = val
        
    return out

def _get_adf_residuals(y: pd.Series, regression: str | None = 'c', **kwargs) -> tuple[np.ndarray, int]:
    """
    Replicate the ADF regression at BIC-selected lag order p and return OLS residuals.

    Regression: Δy_t = α + βt + γy_{t-1} + Σδ_j Δy_{t-j} + ε_t

    Parameters
    ----------
    y : pd.Series
        The level series.
    regression : str
        Deterministic specification: 'n', 'c', 'ct', or 'ctt'.

    Returns
    -------
    residuals : np.ndarray, shape (T-1-p,)
        OLS residuals from the ADF regression.
    p : int
        BIC-selected lag order.
    """
    T = len(y)
    dy = y.diff().dropna().to_numpy()                      # Δy_t, shape (T-1,)
    yarr = y.to_numpy()
    
    p = adfuller(y, **kwargs)[2]    
    n  = T - 1 - p                                         # estimation sample size

    dy_dep  = dy[p:]                                       # Δy_t trimmed to estimation sample
    y_lag   = yarr[p: T-1]                                    # y_{t-1}


    const   = np.ones(n)
    trend   = np.arange(p + 1, T)                          # 1-based trend
    trend2  = np.arange(p + 1, T)**2

    if p > 0:
        lagged_diffs = sm.tsa.tsatools.lagmat(dy,p, trim='both', original='ex')
        
        if regression == 'c':
            Z = np.column_stack([const, y_lag, lagged_diffs])
        elif regression == 'ct':
            Z = np.column_stack([const, trend, y_lag, lagged_diffs])
        elif regression == 'ctt':
            Z = np.column_stack([const, trend, trend2, y_lag, lagged_diffs])
        else:
            Z = np.column_stack([y_lag, lagged_diffs])
    else:
        if regression == 'c':
            Z = np.column_stack([const, y_lag])
        elif regression == 'ct':
            Z = np.column_stack([const, trend, y_lag])
        elif regression == 'ctt':
            Z = np.column_stack([const, trend, trend2, y_lag])
        else:
            Z = np.column_stack([y_lag])

    coeffs, _, _, _ = np.linalg.lstsq(Z, dy_dep, rcond=None)
    residuals = dy_dep - Z @ coeffs

    return residuals, p

def _wild_bootstrap_adf(y: pd.Series, stat: float, resid: np.ndarray,
                         p: int, B: int, regression: str = 'ct',
                         rng: np.random.Generator = None) -> float:
    """
    Wild bootstrap ADF following Cavaliere & Taylor (2008).

    Parameters
    ----------
    y : pd.Series
        Original level series.
    stat : float
        ADF t-statistic on the original series.
    resid : np.ndarray, shape (T-1-p,)
        OLS residuals from the ADF regression on the original series.
    p : int
        Lag order selected by BIC — held fixed across all bootstrap replications.
    B : int
        Number of bootstrap replications.
    regression : str
        Deterministic specification passed to adfuller: 'n', 'c', 'ct', or 'ctt'.
    rng : np.random.Generator
        Random number generator passed from the top-level function.

    Returns
    -------
    p_value : float
        Bootstrap p-value.
    """
    n_resid    = len(resid)
    boot_stats = np.empty(B)

    for b in range(B):
        w        = rng.choice([-1.0, 1.0], size=n_resid)      # Rademacher weights
        eps_star = w * resid                                    # bootstrap innovations

        # cumulate under unit root null: y*_t = sum_{s=1}^{t} eps*_s
        y_star = np.concatenate([[0.0], np.cumsum(eps_star)])  # shape (n_resid+1,)

        # ADF at fixed lag p — autolag=None enforces no lag re-selection
        boot_stat, *_ = adfuller(y_star, maxlag=p, regression=regression, autolag=None)
        boot_stats[b] = boot_stat

    p_value = np.mean(boot_stats <= stat)    # left-tailed: reject for large negative values

    return p_value

def CT_unitroot(y: pd.Series, regression: str = 'ct', B: int = 999,
                     rng: np.random.Generator = None, **kwargs) -> dict:
    """
    Wild bootstrap ADF unit root test of Cavaliere & Taylor (2008).
    Robust to nonstationary volatility (variance breaks, GARCH, heteroskedasticity).

    Parameters
    ----------
    y : pd.Series
        The level series to test.
    regression : str
        Deterministic specification: 'n', 'c', 'ct', or 'ctt'. Default 'ct'.
    B : int
        Number of bootstrap replications. Default 999.
    rng : np.random.Generator
        Random number generator. Must be passed from the main script.
    **kwargs
        Additional arguments passed to adfuller (e.g. maxlag, autolag).

    Returns
    -------
    dict with keys:
        stat      : float  — ADF t-statistic on the original series
        p_value   : float  — bootstrap p-value
        lag       : int    — BIC-selected lag order
        bootstrap : str    — bootstrap scheme label
        regression: str    — deterministic specification used
    """
    if rng is None:
        raise ValueError('rng must be provided. Instantiate with np.random.default_rng(seed) in the main script.')

    if y.isna().any():
        raise ValueError('Series contains NaN values.')

    # step 1 — ADF statistic on original series and residuals
    resid, p = _get_adf_residuals(y, regression=regression, **kwargs)
    stat, *_ = adfuller(y, maxlag=p, regression=regression, autolag=None)

    # step 2 — wild bootstrap p-value
    p_value = _wild_bootstrap_adf(y, stat=stat, resid=resid, p=p,
                                   B=B, regression=regression, rng=rng)

    return {
        'stat'      : stat,
        'p-value'   : p_value,
        'lag'       : p,
        'bootstrap' : 'Rademacher wild bootstrap',
        'regression': regression
    }

