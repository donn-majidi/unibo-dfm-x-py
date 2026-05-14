import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

def ABC_crit(data: pd.DataFrame, kmax: int , nbck: int | None = None, cmax: int | None = 3,
             seed: int | None = 1776, demean: bool | None = False, graph: bool | None = False):
    
    if data.isna().any().any():
        raise ValueError('Dataset includes NaN values.')
        
    if kmax is None:
        raise ValueError('Must provide a value for kmax.')
    
    npace = 1
    step = 500
    T, n  = data.shape
    
    if nbck is None:
        nbck = int(np.floor(n/10))
    
    if demean:
        data = (data - data.mean())/data.std()
        
    ## initialize the Criterion
    
    s = 0
    IC1 = np.zeros(kmax+1)
    #abc = np.zeros((np.floor((nbck+1)//npace), (cmax * step)+1), dtype=int)
    abc = np.zeros(((nbck + 1) // npace, int(cmax * step)), dtype=int)
    rng = np.random.default_rng(seed=seed)
    
    for N in range(n-nbck,n+1,npace):
        perm = rng.permutation(n)
        
        xs = data.iloc[:,perm[:N]]
        _eigvals = np.flip(np.linalg.eigvalsh(xs.cov()))
        
        for k in range(kmax+1):
            IC1[k] = _eigvals[k:N].sum()
        
        p = (N+T)/(N*T) * np.log( (N*T)/(N+T) ) ## Bai and Ng penalty
        T0 = np.arange(kmax + 1) * p
        
        for c in range(1,int(cmax*step)+1):
            cc = c/step
            IC = IC1/N + T0*cc
            rr = np.argmin(IC)
            
            abc[s,c-1] = rr
            
        s += 1
            
    cr = np.arange(1,int(cmax*step)+1)/step
    #ABC = np.array([[kmax,0,0],[0,0,0]])
    
    ABC = np.zeros((cr.shape[0]+1, 4))
    ABC[0,0] = kmax
    
    sabc = np.std(abc, axis=0, ddof=1)
    
    c1 = 1
    for i in range(len(cr)):
        if sabc[i] == 0:
            if abc[-1,i] == ABC[c1-1,0]:
                ABC[c1-1,2] = cr[i]
            else:
                ABC[c1,0] = abc[-1,i]
                ABC[c1,1] = cr[i]
                ABC[c1,2] = cr[i]
                c1 += 1
    
    ABC[:,3] = ABC[:,2] - ABC[:,1]
    ABC = ABC[:c1]
    
    rhat1 = ABC[np.where(ABC[1:, 3] > 0.05)[0] + 1, 0][0]
    rhat2 = ABC[np.where(ABC[1:, 3] > 0.01)[0] + 1, 0][0]
    
    out = [rhat1, rhat2]
    
    return out


    
    
            
    
    
            
            
            
        
        
        
        
    
    
    
    