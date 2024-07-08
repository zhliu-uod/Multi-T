import numpy as np
from utils import dist, normIt
from scipy import stats 
from numpy import polyfit, poly1d

## Shell-Renorm
def estShell(data):
    m_ = np.mean(data, axis=0, keepdims=True)
    d = np.linalg.norm(data - m_, axis=1)
    var = np.mean(d)

    err = np.absolute(d-var)
    MAD =  np.median(err)
    eSig = 1.4826*MAD

    return m_, var, eSig

def projectMean(data, m, var):
    d = np.linalg.norm(data - m, axis=1)
    err = d-var
    return err

def robustMean(featTrain, globalMean, thres=2, numIter=10):
    
    feat = normIt(featTrain, globalMean)
    m_, var, eSig = estShell(feat)
    err = projectMean(feat, m_, var)
    mask = err>eSig*thres
    meanInlier = np.mean(featTrain[mask,:], axis=0)
    meanOutlier = np.mean(featTrain[~mask,:], axis=0)
    globalMean = (meanInlier + meanOutlier)/2

    for i in range(numIter):
        feat = normIt(featTrain, globalMean)
        m_, var, eSig = estShell(feat[mask])
        err = projectMean(feat, m_, var)
        mask = err>eSig*thres

        meanInlier = np.mean(featTrain[mask,:], axis=0)
        meanOutlier = np.mean(featTrain[~mask,:], axis=0)
        globalMean = (meanInlier + meanOutlier)/2
        #newMean = meanOutlier
    return err

## Multi-T
def _poly_fit(score):
    degree = 1 # linear regressor
    x = np.arange(len(score))
    poly_f = poly1d(polyfit(x, sorted(score), 1))
    thres = poly_f(np.hstack(np.array(np.where(poly_f(x)>sorted(score))))[-1])
    num_normal = np.sum(score<thres)
    return num_normal, thres

def _three_sigma_k(score, num_normal, k):
    # By default: k = 3
    m = np.mean(score[np.argsort(score)[:num_normal]])
    std = np.std(score[np.argsort(score)[:num_normal]])
    thres = m+k*std
    return thres
 
def _robust_thres(data):
    # l2-norm
    data_norm = normIt(data) # ergodic-set normalization
    score = dist(data_norm) # euclidean distance (l2 norm)
    
    # Shell Renormalization
    globalMean = np.mean(data_norm, axis = 0)
    score_re = robustMean(data_norm, globalMean, thres=1, numIter=10)
    
    # ranking-lists
    sort_list_l2 = np.argsort(score) 
    sort_list_re = np.argsort(score_re) 
    
    # ranking-list's similarity
    spearmanr_simi = stats.spearmanr(sort_list_l2, sort_list_re).statistic
    
    # poly_nominal fitting 
    normal_num, _ = _poly_fit(score)
    thres = _three_sigma_k(score, normal_num, 3)
    if spearmanr_simi >= 0.1:
        hard_thres = thres
    else: 
        hard_thres = np.mean(score)+3*np.std(score) 
    
    for _ in range(20):
        normal_score = score[score <= thres]
        normal_num, soft_thres = self._poly_fit(normal_score)
        
        thres_last = thres
        thres = self._three_sigma_k(score, normal_num, 3)
        if thres == thres_last:
            break
        
    if spearmanr_simi >= 0.3:
        hard_thres = thres
    
    return hard_thres, soft_thres 
