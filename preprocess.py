
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fastdtw import fastdtw # dynamic time warping distance
from scipy.spatial.distance import euclidean # needed for fastdtw package
from itertools import combinations


''' The df that is imported in the preprocess function has the following structure
    each column is a datapoint with the progression 
    t1rep1, t1rep2, t1rep3, t1Mrep1, t1Mrep2, t1Mrep3, t2rep1, ..., t12Mrep3 '''

def get_groups_from_df(data,labels): 
    ''' The df that is imported has the following structure
    each column is a datapoint with the progression 
    1rep1, 1rep2, 1rep3, 1Mrep1, 1Mrep2, 1Mrep3, 2rep1, ..., 12Mrep3
    This functions builds the two matrices (one per group)
    1rep1, 1rep2, 1rep3, ..., 12rep1, 12rep2, 12rep3
    1Mrep1, 1Mrep3, 1Mrep3, ..., 12Mrep1, 12Mrep2, 12Mrep3 '''
    
    
    data_c = np.zeros([data.shape[0],int(data.shape[1]/2)]) # c for control
    data_t = np.zeros([data.shape[0],int(data.shape[1]/2)]) # t for treatment
    c = 0
    ct = 0
    for i in range(0,data.shape[1]):
        if 'M' not in labels[i]: # 'M' stands for malathion (the treatment of group 2) in the labels
            data_c[:,c] = data[:,i]
            c += 1
        elif 'M' in labels[i]:
            data_t[:,ct] = data[:,i]
            ct += 1
    return data_c, data_t

def get_reps(reps,ntimepts):
    ''' get the columns inds of the replicates that you want to keep'''
    nreps = 3
    allinds = set(list(range(0,ntimepts*nreps)))
    if reps == [0]:
        keepers = list(allinds - set(list(range(1,ntimepts*nreps,3))) - set(list(range(2,ntimepts*nreps,3))))
    elif reps == [1]:
        keepers = list(allinds - set(list(range(0,ntimepts*nreps,3))) - set(list(range(2,ntimepts*nreps,3))))
    elif reps == [2]:
        keepers = list(allinds - set(list(range(0,ntimepts*nreps,3))) - set(list(range(1,ntimepts*nreps,3))))
    elif reps == [0,1]: # set subtract column inds for rep3 from allinds
        keepers = list(allinds - set(list(range(2,ntimepts*nreps,3)))) # column inds for rep1 and rep2
    elif reps == [0,2]: # set subtract column inds for rep2 from allinds
        keepers = list(allinds - set(list(range(1,ntimepts*nreps,3)))) # column inds for rep1 and rep3
    elif reps == [1,2]: # set subtract column inds for rep1 from allinds
        keepers = list(allinds - set(list(range(0,ntimepts*nreps,3))))
    elif reps == [0,1,2]:
        keepers = list(allinds)
    keepers.sort()
    return keepers

def put_groups_in_3D(data,nTraj,nT):
    '''Data from each trajectory (replicate) is placed in a new 2d array which is appended to one 3d array of 
    dimension n x m x r. n is number of genes, m is number of timepoints, r is number of replicates.'''
    X = np.zeros((data.shape[0],nT,nTraj))
    reps = list(range(0,nTraj))
    for i in reps:
        X[:,:,i] = data[:,get_reps([i],nT)]
    return X

def cv_filter(data,cv_cutoff=0.148,violationAllowance=9):
    '''data is of shape n x m x r'''
    mean = np.mean(data,axis=2)
    stdev = np.std(data,axis=2)
    cv = stdev/mean # cv[i,j] gives the cv of the ith gene at the jth timepoint, calculated over all replicates
    cv_thresh = (cv < cv_cutoff)*cv
    cv_thresh[cv_thresh > 0.0] = 1
    cv_thresh_sum = np.sum(cv_thresh,axis=1)
    keepers = list((np.nonzero((cv_thresh_sum >= violationAllowance) * cv_thresh_sum))[0])
    keepers.sort()
    return keepers

def dtw_filter(data,thresh=0.00095):
    '''Dynamic time warping distance between a gene's time series measured over two replicates'''
    keepers = []
    for i in range(0,len(data)):
        distance,path = fastdtw(data[i,:,0],data[i,:,1],dist=euclidean)
        distance /= (np.linalg.norm(data[i,:,0],ord=2)*np.linalg.norm(data[i,:,1],ord=2))
        if distance <= thresh: 
            keepers.append(i)
    keepers.sort()
    return keepers

def num_high_expression(data,nT,nTraj,thresh=100):
    '''count how many genes, after filtering, have mean over time and over replicates > thresh'''
    means = np.mean(data.reshape(len(data),nT*nTraj),axis=1)
    return len(data[means>thresh])

def scaler(data,method=MinMaxScaler):
    '''transforms data of shape n x m x r such that each feature's time series has range of 1.0 to 2.0 '''
    data_normed = np.zeros(data.shape)
    scalerList = []
    for i in range(0,data_normed.shape[2]): 
        scaler = MinMaxScaler().fit(data[:,:,i].T)
        scalerList.append(scaler)
        data_normed[:,:,i] = (scaler.transform(data[:,:,i].T)).T 
    return data_normed + 1.0 # this affine term is to lift the data from 0.0

def apply_filter_before_backsub(data_c,data_t,reps,filterMethod): 
    repList = []
    for repcomb in combinations(reps,2): # get lists of all possible length 2 combinations of the replicates being used
        repList.append(list(repcomb)) # if reps=[0,1,2], repList = [[0,1],[0,2],[1,2]]
    if filterMethod == 'CV': # filter based on coefficient of variation
        keepers_c,keepers_t = cv_filter(data_c),cv_filter(data_t) # do we want to remove the first and last tps?
        keepers_repr = list(set(keepers_c)&set(keepers_t))
    elif filterMethod == 'DTW': # filter based on dynamic time warping distance. done in pairs of replicates
        keepers_c_r = []
        if len(reps) == 3: 
            for these_reps in repList:
                keepers_c_r.append(dtw_filter(data_c[:,:,these_reps]))
                keepers_c_r.append(dtw_filter(data_t[:,:,these_reps]))
            keepers_c_r = [set(k) for k in keepers_c_r]
            keepers_repr = list(set.intersection(*keepers_c_r))
        elif len(reps) == 2: 
            keepers_c_r.append(dtw_filter(data_c))
            keepers_c_r.append(dtw_filter(data_t))
            keepers_c_r = [set(k) for k in keepers_c_r]
            keepers_repr = list(set.intersection(*keepers_c_r))
    print('Keeping',len(keepers_repr),'genes out of',len(data_c))
    return keepers_repr

def apply_filter_after_backsub(data,reps,filterMethod): 
    repList = []
    for repcomb in combinations(reps,2): # get lists of all possible length 2 combinations of the replicates being used
        repList.append(list(repcomb)) # if reps=[0,1,2], repList = [[0,1],[0,2],[1,2]]
    if filterMethod == 'CV': # filter based on coefficient of variation
        keepers_repr = cv_filter(data)
    elif filterMethod == 'DTW': # filter based on dynamic time warping distance. done in pairs of replicates
        dtwThresh = 0.073 # this threshold removes all but 506 genes. 
        keepers_c_r = []
        if len(reps) == 3: 
            for these_reps in repList:
                keepers_c_r.append(dtw_filter(data_c[:,:,these_reps]))
                keepers_c_r.append(dtw_filter(data_t[:,:,these_reps]))
            keepers_c_r = [set(k) for k in keepers_c_r]
            keepers_repr = list(set.intersection(*keepers_c_r))
        elif len(reps) == 2: 
            keepers_c_r.append(dtw_filter(data_c))
            keepers_c_r.append(dtw_filter(data_t))
            keepers_c_r = [set(k) for k in keepers_c_r]
            keepers_repr = list(set.intersection(*keepers_c_r))
    print('Keeping',len(keepers_repr),'genes out of',len(data))
    return keepers_repr

def preprocess(datadir,reps,ntimepts,Norm=False,Filter=True,filterMethod='CV',filterB4BackSub=True):
    start_time = time.time()
    print('---------Preprocessing replicates',reps,'---------')

    df = pd.read_csv(datadir)
    nreps = 3 # number of replicates in the P. fluorescens malathion dataset
    sampleLabels = list(df.columns[1:])
    transcriptIDs = list(df.iloc[:,0])
    df = df.iloc[:,1:] # first column contains transcriptIDs

    data_c, data_t = get_groups_from_df(np.array(df),sampleLabels) 
    data_c, data_t = put_groups_in_3D(data_c,nreps,ntimepts), put_groups_in_3D(data_t,nreps,ntimepts) # there are 3 replicates in dataset, which explains the hardcoded 3

    # timepoints and reps to use for parameter fitting
    print(data_c.shape,data_t.shape)
    data_c = data_c[:,2:-1,reps] # not going to use the first and last timepoints due to anomalous data
    data_t = data_t[:,2:-1,reps] 
    newntimepts = data_c.shape[1]

    # filter nonreproducible genes before back sub based on chosen criteria (DTW, CV, mean distance) 
    if Filter and filterB4BackSub: 
        keepers_repr = apply_filter_before_backsub(data_c,data_t,reps,filterMethod)

    # normalize each gene's time series, not each snapshot!
    if Norm: 
        data_c, data_t = scaler(data_c,method=MinMaxScaler), scaler(data_c,method=MinMaxScaler)

    # subtract the control from the treatment condition (background subtraction)
    X = np.maximum(data_t - data_c,0)

    # filter nonreproducible genes after background subtraction on chosen criteria (DTW, CV, mean distance) 
    if Filter and not filterB4BackSub:
        keepers_repr = apply_filter_after_backsub(data,reps,filterMethod)

    if Filter: 
        X = X[keepers_repr]

    print((time.time() - start_time)/60, 'minutes')

    return X, transcriptIDs, keepers_repr, newntimepts






    
