import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(-1, './src/')

from fred_transform import fred_qd_transform
from diagnostics import adf_wrapper
from diagnostics import kpss_wrapper
from diagnostics import CT_unitroot
from ABC_crit import ABC_crit

plt.rc('figure', figsize=(18,10))
sns.set_style('darkgrid')

rng = np.random.default_rng(1776)

df = pd.read_csv('./data/2026-04-QD.csv')

fred_df = fred_qd_transform(df)
fred_df = fred_df.iloc[1:,:]
fred_df = fred_df.iloc[:-1,:]
fred_df = fred_df.dropna(axis=0)

nonstat_names = []
nonstat_names2 = []
nonstat_names3 = []

for i in fred_df.columns:
    adf_results = adf_wrapper(fred_df[f'{i}'])
#    kpss_results = kpss_wrapper(fred_df[f'{i}'])
#    CT_results = CT_unitroot(fred_df[i], rng=rng)
    
    if adf_results['p-value'] > 0.05:
        nonstat_names.append(fred_df[f'{i}'].name)
        
#    if CT_results['p-value'] > 0.05:
#        nonstat_names2.append(i)
        
#    if kpss_results['p-value bound'] < 0.05:
#        nonstat_names3.append(fred_df[f'{i}'].name)
        

#fred_df.drop(columns=nonstat_names, inplace=True)   

fig, ax = plt.subplots()

rhat1, rhat2, ax = ABC_crit(fred_df, 10, demean=True, ax=ax)

plt.show()



