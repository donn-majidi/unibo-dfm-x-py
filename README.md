# unibo-dfm-x-py
A Python framework for macroeconomic forecasting and nowcasting with dynamic factor models (DFMs).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Modules](#modules)
- [Workflow](#workflow)
- [Usage Example](#usage-example)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Overview

## Installation

## Modules

### `fred_transform`
This function transforms each of the series in the FRED QD and MD datasets according to their transformation codes and returns a clean pandas dataframe containing the transformed series indexed by time.
- The function takes as input the raw dataset downloaded from the FRED database and loaded via pandas.
- Additionally, the function takes a two character string input ('QD' or 'MD') to determine the frequency of the input data.

```python
import pandas as pd
from src.fred_transform import fred_transform

## Quarterly data
fred_qd = pd.read_csv('./data/2026-04-QD.csv')
fred_qd_tr = fred_transform(fred_qd, 'QD')

## Monthly data
fred_md = pd.read_csv('./data/2026-04-MD.csv')
fred_md_tr = fred_transform(fred_md, 'MD')
```

### `ABC_crit`
This function implements the information criterion of Alessi, Barigozzi and Capasso (2010).
- The input data has to be a pandas dataframe. It should not contain missing values or else the function throws out an error.
- The upper bound for the number of factors `kmax` is a required input and must provided.
- If `ax` is provided, the function also plots the IC on the same canvas on `ax`.

```python
import matplotlib.pyplot as plt
from src.ABC_crit import ABC_crit

## load the data
T, n = data.shape
## Bai and Ng (2002) upper bound for kmax
kf = int((min(n,T)/100)**0.25)
kmax = 8 * kf if kf > 0 else 8

fig, ax = plt.subplots()
rhat1, rhat2, ax = ABC_crit(data, kmax=kmax, ax=ax, demean=True)
```

## Workflow
The full estimation pipeline proceeds in five stages:
 
```
FRED-QD or FRED-MD raw data
      │
      ▼
1. fred_transform           — apply stationarity transformations per tcode
      │
      ▼
2. ABC_crit                 — estimate number of common factors r
      │
      ▼
3. SVD / PCA                — initialize loadings Λ and factors F via SVD of standardized data
      │
      ▼
4. VAR lag selection        — AIC on VAR(F), confirmed via likelihood ratio test
      │
      ▼
5. DynamicFactorMQ          — state space estimation via Kalman filter + EM algorithm
```
 
---
## Usage Example

## Requirements

## References

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
