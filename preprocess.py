
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fastdtw import fastdtw # dynamic time warping distance
from scipy.spatial.distance import euclidean # needed for fastdtw package


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

def dtw_filter(data,reps,thresh=0.00095):
    '''Dynamic time warping distance between a gene's time series measured over two replicates'''
    keepers = []
    for i in range(0,len(data)):
        distance,path = fastdtw(data[i,:,reps[0]],data[i,:,reps[1]],dist=euclidean)
        distance /= (np.linalg.norm(data[i,:,reps[0]],ord=2)*np.linalg.norm(data[i,:,reps[1]],ord=2))
        if distance <= thresh: 
            keepers.append(i)
    keepers.sort()
    return keepers

def mean_distance_filter(data,reps,thresh=0.0142):
    ''' mod(||x_i^(r1)-0.5*(x_i^(r1)+x_i^(r2))|| / ||x_i^(r1)|| -
            ||x_i^(r2)-0.5*(x_i^(r2)+x_i^(r1))|| / ||x_i^(r2)||)
    x_i is the time trace of gene i, and data is of shape n x m x r
    reps is a list of two indices corresponding to replicates to be compared'''
    dist1 = np.linalg.norm(data[:,:,reps[0]] - 0.5*(data[:,:,reps[0]] + data[:,:,reps[1]]),ord=2,axis=1) / \
                np.linalg.norm(data[:,:,reps[0]],ord=2,axis=1)
    dist2 = np.linalg.norm(data[:,:,reps[1]] - 0.5*(data[:,:,reps[1]] + data[:,:,reps[0]]),ord=2,axis=1) / \
                np.linalg.norm(data[:,:,reps[1]],ord=2,axis=1)
    dist = np.abs(dist1 - dist2)
    keepers_tmp = (dist < thresh) * dist
    keepers = list(np.nonzero(keepers_tmp)[0])
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

def preprocess(datadir,reps,ntimepts,Norm=False,Filter=True,filterMethod='CV'):
    start_time = time.time()
    print('---------Preprocessing replicates',reps,'---------')

    df = pd.read_csv(datadir)
    nreps = len(reps)
    sampleLabels = list(df.columns[1:])
    transcriptIDs = list(df.iloc[:,0])
    df = df.iloc[:,1:] # first column contains transcriptIDs

    data_c, data_t = get_groups_from_df(np.array(df),sampleLabels) 
    data_c, data_t = put_groups_in_3D(data_c,nreps,ntimepts), put_groups_in_3D(data_t,nreps,ntimepts) 

    # filter nonreproducible genes based on chosen criteria (DTW, CV, cosine similarity) 
    if Filter: # not going to use the first and last timepoints due to anomalous data
        if filterMethod == 'CV': # filter based on coefficient of variation
            keepers_c,keepers_t = cv_filter(data_c[:,1:-1]),cv_filter(data_t[:,1:-1]) # do we want to remove the first and last tps?
            keepers_repr = list(set(keepers_c)&set(keepers_t))
        elif filterMethod == 'MD': # filter based on distance from mean. done in pairs of replicates
            repList = [0,1]
            keepers_c1,keepers_t1 = mean_distance_filter(data_c[:,1:-1],repList),mean_distance_filter(data_t[:,1:-1],repList)
            repList = [0,2]
            keepers_c2,keepers_t2 = mean_distance_filter(data_c[:,1:-1],repList),mean_distance_filter(data_t[:,1:-1],repList)
            repList = [1,2]
            keepers_c3,keepers_t3 = mean_distance_filter(data_c[:,1:-1],repList),mean_distance_filter(data_t[:,1:-1],repList)
            keepers_repr = list(set(keepers_c1)&set(keepers_t1)&set(keepers_c2)&set(keepers_t2)&set(keepers_c3)&set(keepers_t3))
        elif filterMethod == 'DTW': # filter based on dynamic time warping distance. done in pairs of replicates
            repList = [0,1]
            keepers_c1,keepers_t1 = dtw_filter(data_c[:,1:-1],repList),dtw_filter(data_t[:,1:-1],repList)
            repList = [0,2]
            keepers_c2,keepers_t2 = dtw_filter(data_c[:,1:-1],repList),dtw_filter(data_t[:,1:-1],repList)
            repList = [1,2]
            keepers_c3,keepers_t3 = dtw_filter(data_c[:,1:-1],repList),dtw_filter(data_t[:,1:-1],repList)
            keepers_repr = list(set(keepers_c1)&set(keepers_t1)&set(keepers_c2)&set(keepers_t2)&set(keepers_c3)&set(keepers_t3))
        print('Keeping',len(keepers_repr),'genes out of',len(df))
        high_expression_c = num_high_expression(data_c[keepers_repr],ntimepts,nreps)
        high_expression_t = num_high_expression(data_t[keepers_repr],ntimepts,nreps)
        print('Number of high expression genes (mean > 100) in control:',high_expression_c,', and in treatment:',high_expression_t)

    data_c = data_c[:,1:-1] 
    data_t = data_t[:,1:-1] # starting condition is the timepoint after malathion was introduced.
    newntimepts = data_c.shape[1]

    # normalize each gene's time series, not each snapshot!
    if Norm: 
        data_c, data_t = scaler(data_c,method=MinMaxScaler), scaler(data_c,method=MinMaxScaler)

    # subtract the control from the treatment condition
    X = np.maximum(data_t - data_c,0)

    # filter out genes which have mean background subtracted expression less than 5
    # mu = (np.mean(np.mean(X,axis=2),axis=1))
    # keepers_mu = list(np.nonzero((mu>5)*mu)[0])

    # keepers = list(set(keepers_mu)&set(keepers_repr))

    X = X[keepers_repr]

    print((time.time() - start_time)/60, 'minutes')

    return X, transcriptIDs, keepers_repr, newntimepts






    
