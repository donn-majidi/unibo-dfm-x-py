import pandas as pd

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
    
