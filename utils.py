import numpy as np  
 

def normIt(data, m=None):
    nData = data.copy()
    if m is None:
        m = np.mean(nData)
    nData = nData - m
    nData = nData / np.linalg.norm(nData, axis =1, keepdims=True)
    return nData 

def dist(feats, m = None):
    if m is None:
        m = np.mean(feats, axis = 0)
    d = np.linalg.norm(feats-m, axis = 1)**2
    return d
 