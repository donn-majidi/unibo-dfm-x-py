import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(-1, './src')

from fred_transform import fred_qd_transform
from diagnostics import adf_wrapper
from diagnostics import kpss_wrapper
from ABC_crit import ABC_crit

df = pd.read_csv('./data/2026-04-QD.csv')

fred_df = fred_qd_transform(df)
fred_df = fred_df.iloc[1:,:]
fred_df = fred_df.iloc[:-1,:]
fred_df = fred_df.dropna(axis=1)

nonstat_names = []
nonstat_names2 = []

for i in fred_df.columns:
    adf_results = adf_wrapper(fred_df[f'{i}'])
    kpss_results = kpss_wrapper(fred_df[f'{i}'])
    if adf_results['p-value'] > 0.05:
        nonstat_names.append(fred_df[f'{i}'].name)
        
    if kpss_results['p-value bound'] < 0.05:
        nonstat_names2.append(fred_df[f'{i}'].name)
        

fred_df.drop(columns=nonstat_names, inplace=True)   

rhat = ABC_crit(fred_df, 8, demean=True)
 
 


