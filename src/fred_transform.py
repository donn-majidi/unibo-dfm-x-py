import numpy as np
import pandas as pd

__transform_codes__ = [1,2,3,4,5,6,7]

def fred_transform (data: pd.Series, transform_code: int):
    """
    This function transforms the input data to achieve stationarity.
    Transformations are based on integer codes which determin the
    type of transformation.
    
    transform_code  == 1 -> No transformation.
                    == 2 -> First difference.
                    == 3 -> Second order difference.
                    == 4 -> Log transform.
                    == 5 -> Log difference.
                    == 6 -> Second order log difference.
                    == 7 -> Relative difference.
                    
    [Reference: FRED-QD: A Quarterly Database for Macroeconomic Research McCracken and Ng, 2020]

    Parameters
    ----------
    data : pd.Series
        Univariate input time-series.
    transform_code : int
        Transformation code.

    Returns
    -------
    pd.Series.
    
     
    Raises
    ------
    ValueError
        If transformation codes are invalid.

    """
    
    if transform_code not in __transform_codes__:
        raise ValueError(f"Transformation code {transform_code} is invalid. Please choose from {__transform_codes__}")
        
    match transform_code:
        case 1:
            data_tr = data
            
        case 2:
            data_tr = data.diff()
            
        case 3:
            data_tr = data.diff().diff()
            
        case 4:
            data_tr = np.log(data)
            
        case 5:
            data_tr = np.log(data).diff()
            
        case 6:
            data_tr = np.log(data).diff().diff()
            
        case 7:
            data_tr = data/data.shift(1) - 1
            
    return data_tr


def fred_qd_transform(data: pd.DataFrame):
    """
    This function transforms the input data to achieve stationarity.
    Transformations are based on integer codes which determin the
    type of transformation.
    
    transform_code  == 1 -> No transformation.
                    == 2 -> First difference.
                    == 3 -> Second order difference.
                    == 4 -> Log transform.
                    == 5 -> Log difference.
                    == 6 -> Second order log difference.
                    == 7 -> Relative difference.
                    
    [Reference: FRED-QD: A Quarterly Database for Macroeconomic Research McCracken and Ng, 2020]

    Parameters
    ----------
    data : pd.DataFrame
        Raw FRED-QD data input.

    Returns
    -------
    Transformed series in a pd.DataFrame object.

    """
    
    transformed = []
    
    for i in range(1,len(data.columns)):
        input_series = data.iloc[2:,i]
        transform_code = data.iloc[1,i]
        input_dates = data.iloc[2:,0]
        
        match transform_code:
            case 1:
                _input_series_tr = input_series
                
            case 2:
                _input_series_tr = input_series.diff()
                
            case 3:
                _input_series_tr = input_series.diff().diff()
                
            case 4:
                _input_series_tr = np.log(input_series)
                
            case 5:
                _input_series_tr = np.log(input_series).diff()
                
            case 6:
                _input_series_tr = np.log(input_series).diff().diff()
                
            case 7:
                _input_series_tr = input_series/input_series.shift(1) - 1
            
            
        _input_series_tr.name = data.columns[i]
        _input_series_tr.index = pd.to_datetime(input_dates, format="%m/%d/%Y", errors='coerce')
        
        transformed.append(_input_series_tr)
        
    transformed_df = pd.concat(transformed, axis=1, join='outer')
    
    return transformed_df

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            