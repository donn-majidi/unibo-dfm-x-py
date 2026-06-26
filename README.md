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
### `ABC_crit`

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
