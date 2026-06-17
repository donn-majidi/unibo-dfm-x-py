import scipy
import numpy as np
import pandas as pd
import sys
sys.path.insert(-1, './src/')

import matplotlib.pyplot as plt
import seaborn as sns
from ABC_crit import ABC_crit
from fred_transform import fred_qd_transform
from statsmodels.tsa.stattools import adfuller, acf, pacf


plt.rc('figure', figsize=(18,10))
sns.set_style('darkgrid')

## Define default_rng with seed for reproducibility
seed = 1776
rng = np.random.default_rng(seed)

"""
1. Load and initialize the data.
    Data source is FredQD Quarterly data base.

"""

## Load data and apply fred transforms
df = pd.read_csv('./data/2026-04-QD.csv')
## Often for some variables the observation corresponding to the last observation date is not yet recorded
df = df.iloc[:-1,:]
fred_df = fred_qd_transform(df)

## The first two observations will be lost due to the stationary I(2) transforms
fred_df = fred_df.iloc[2:,:]

## Choose how to handle missing data
ls_missing = fred_df.isna().any()
ls_missing = ls_missing.loc[ls_missing == True]
ls_missing.shape

series_init = dict()

## To get a sense of the starting date for each series
for key, _ in ls_missing.items():
    fr_index = fred_df[key].first_valid_index()
    series_init[key] = fr_index
    
## Starting dates vary from one variable to another
## To eliminate the NaNs from the data set, we can move the starting date to a future point in time
## and then remove the remaining columns for which the data set still contains missing values.
## Setting the starting point to 1971Q3 and then removing the remaining columns with NaNs seems to be a good compromise.

fred_df = fred_df.loc['1971-06-01':,:]
fred_df = fred_df.dropna(axis=1)

## Some series are still non-stationary even after transformations
## Need to identify and remove them from the data set.

nonstat_names = []
for c in fred_df.columns:
    adf_pval = adfuller(fred_df[c].dropna(), autolag='AIC', regression='c')[1]
    if adf_pval > 0.05:
        nonstat_names.append(c)
        

fred_df = fred_df.drop(columns=nonstat_names)


"""
2. Factor Analysis via Static Methods
    2.1 Estiamte the number of common factors via the ABC criterion.
    2.2 Obtain estimates of the loadings and the factors via PCA and linear projection under the identification scheme:
        (F.T @ F)/T = I_r
    2.4 Box-Jenkins paradigm to identify the order of the VAR process for modeling the dynamics of the common factors.

"""
T, n = fred_df.shape
## Bai and Ng (2002) upper bound for kmax
kf = int((min(n,T)/100)**0.25)
kmax = 8 * kf if kf > 0 else 8

fig, ax = plt.subplots()
rhat1, rhat2, plot = ABC_crit(fred_df, kmax, seed=seed, ax=ax, demean=True)
plt.show()
## rhat1 and rhat2 are the estimated number of common factors at 5 and 1 percent thresholds.

## To proceed, we first standardize the data
fred_df = (fred_df - fred_df.mean())/fred_df.std()

## PC Analysis via SVD of the data matrix
U, s, Vh = scipy.linalg.svd(fred_df, full_matrices=False)
V = Vh.T
S = np.diag(s)

## Visualize the scree plot
p = sns.scatterplot(data=s/T)
p.set(xlabel='Component Index',
      ylabel='Singular Value (Scaled by T)',
      title='Scree Plot')

## Principal components are obtained as the linear transformation of the data via the transpose of the matrix of right singular vectors
X_PC = Vh @ fred_df.transpose()

"""
Under the identification assumption E[ff'] = I, we have:
                                      
        X = F Lambda.T + Xi -> (X.T @ X)/T = Lambda ( F.T @ F )/T Lambda.T + (Xi.T @ Xi)/T
            
    From the SVD of the data matrix we have:
            
        X = U @ S @ V.T -> 
                (X.T @ X)/T = V @ (S^2)/T @ V.T
                            = V_x @ (S^2_x)/T @ V_x.T + V_(-x) @ (S^2_(-x))/T @ V_(-x).T
                            
    Where S_x and V_x are the matrix of singular values and the corresponding matrix of the right singular vectors of the x largest singular values
    
    Trivially, the PCA estiamte of the matrix of loadings is obtained as:
        Lambda_hat = (V_x @ S_x)/sqrt(T)
                                      
"""
V_x = V[:,:5]
S_x = S[:5,:5]

Lambda_hat = V_x @ S_x / np.sqrt(T)

## The factors are obtained via the linear projection of the data onto the range space of the matrix of loadings
"""
    X = F Lambda.T + Xi
    Objective -> minimize Tr(Xi @ Xi.T) w.r.t F <~~ Minimize the sum of residuals over N and T
    
    The minimization problem follows:
        (X - F Lambda.T) (X - F Lambda.T).T = XX.T - F Lambda.T X.T - X Lambda F.T + F Lambda.T Lambda F.T
    
    And the minimzer is found to be the linear projection:
        F_hat = X Lambda_hat (Lambda_hat.T Lambda_hat)^(-1)
        
    Since Lambda_hat is already given in terms of the singular values and right singular vectors we can deduce:
        F_hat = X V_x S_x/T (S_x/sqrt(T) V_x V_x.T S_x/sqrt(T))^(-1) = X V_x (S_x/sqrt(T))^(-1)

"""
F_hat = fred_df @ V_x @ np.linalg.inv(S_x)/np.sqrt(T)


"""
3. State Space Representation and Estimation of the DFM via The Kalman Filter and The EM Algorithm
"""


"""
4. Post-Estimation Analysis and Diagnostic Tests
"""

"""
5. Forecasting via The Common Component
"""




    
    
    
    