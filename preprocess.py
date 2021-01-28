
import pandas as pd
import numpy as np
import pickle
import time

''' The df that is imported in the preprocess function has the following structure
    each column is a datapoint with the progression 
    1rep1, 1rep2, 1rep3, 1Mrep1, 1Mrep2, 1Mrep3, 2rep1, ..., 12Mrep3 '''

def get_groups_from_df(data,labels): 
    ''' The df that is imported has the following structure
    each column is a datapoint with the progression 
    1rep1, 1rep2, 1rep3, 1Mrep1, 1Mrep2, 1Mrep3, 2rep1, ..., 12Mrep3
    This functions builds the two matrices (one per group)
    1rep1, 1rep2, 1rep3, ..., 12rep1, 12rep2, 12rep3
    1Mrep1, 1Mrep3, 1Mrep3, ..., 12Mrep1, 12Mrep2, 12Mrep3 '''
    
    # will be easier to compute statistics after this permutation
    
    data_c = np.zeros([data.shape[0],int(data.shape[1]/2)]) # c for control
    data_i = np.zeros([data.shape[0],int(data.shape[1]/2)]) # i for input
    c = 0
    ci = 0
    for i in range(0,data.shape[1]):
        if 'M' not in labels[i]: # 'M' stands for malathion (the input to group 2) in the labels
            data_c[:,c] = data[:,i]
            c += 1
        elif 'M' in labels[i]:
            data_i[:,ci] = data[:,i]
            ci += 1
    return data_c, data_i

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

def mean_filter(control,treatment,mean_cutoff=0.25):
    ''' filter genes with low expression '''
    data = np.mean(np.maximum(treatment - control,0.0),axis=1)
    keepers_mean = []
    for i in range(len(data)):
        if (data[i] >= mean_cutoff):
            keepers_mean.append(i)
    return keepers_mean

def cv_filter(mean,stdev,cv_cutoff=0.2):
    ''' filter genes with high cv '''
    allowCutoffViolation_thresh = int(mean.shape[1]/2)
    cv = stdev/mean # cv[i,j] gives the cv of the ith gene at the jth timepoint, calculated over all replicates
    cv_thresh = (cv < cv_cutoff)*cv
    cv_thresh[cv_thresh > 0.0] = 1
    cv_thresh_sum = np.sum(cv_thresh,axis=1)
    keepers_cv = list((np.nonzero((cv_thresh_sum >= allowCutoffViolation_thresh) * cv_thresh_sum))[0])
    return keepers_cv

def get_global_from_group(data,nTraj,nT):
    ''' This takes as input, the outputs of get_groups_from_df()
    This function outputs the following structured matrix:
    1rep1 2rep1 ... 12rep1 1rep2 ... 12rep3'''
    # nt is the number of timepoints per trajectory
    # ntraj is the number of trajectories
    
    # this is to be input into get_snapshots_from_global 
    
    X = np.zeros(data.shape)
    for i in range(0,nTraj):
        idx = np.arange(i,data.shape[1],nTraj)
        X[:,i*nT:i*nT+nT] = data[:,idx]
    return X

def get_rep_stats(data,nTraj,nT):
    '''get replicate statistics
    - with data structured as t1rep1,t1rep2,t1rep3...t12rep3
    - nTraj - # of trajectories (replicates), nT - # of timepts per trajectory'''
    
    mean = np.zeros([data.shape[0],int(data.shape[1]/nTraj)])
    stdev = np.zeros([data.shape[0],int(data.shape[1]/nTraj)])
    for i in range(0,nT):
        mean[:,i] = np.mean(data[:,i*nTraj:i*nTraj+nTraj],axis=1)
        stdev[:,i] = np.std(data[:,i*nTraj:i*nTraj+nTraj],axis=1)
    return mean,stdev

def preprocess(datadir,reps,ntimepts,ntimeptsModel,MEANCUTOFF=0.12,CVCUTOFF=0.1,log=True,norm=False,Filter=True,save_data=False):
    start_time = time.time()
    print('---------Preprocessing replicates',reps,'---------')

    df = pd.read_csv(datadir)
    nreps = len(reps)
    sampleLabels = list(df.columns[1:])
    geneIDs = list(df.iloc[:,0])
    if log: 
        df = np.log2(df.iloc[:,1:]) # first column contains geneIDs
    else: 
        df = df.iloc[:,1:]

    data_c, data_t = get_groups_from_df(np.array(df),sampleLabels) # easy to use to calc stats

    if Filter:
        all_reps = [0,1,2] # filter genes based on all replicate statistics. 
    else:
        all_reps = reps

    # get replicate statistics
    mean_c, stdev_c = get_rep_stats(data_c,len(all_reps),ntimepts)
    mean_t, stdev_t = get_rep_stats(data_t,len(all_reps),ntimepts)

    # filter low expression and noisy genes
    keepers_mean = mean_filter(data_c,data_t,mean_cutoff=MEANCUTOFF)
    keepers_cv_c, keepers_cv_t =  cv_filter(mean_c,stdev_c,cv_cutoff=CVCUTOFF),cv_filter(mean_t,stdev_t,cv_cutoff=CVCUTOFF)
    keepers_cv = list(set(keepers_cv_c) & set(keepers_cv_t))
    keepers = list(set(keepers_mean) & set(keepers_cv))
    if Filter:
        print(len(keepers), 'genes with CV less than', CVCUTOFF, 'and expression greater than', MEANCUTOFF, '(across all replicates)')

        # filter for the desired replicates
        keep_reps = get_reps(reps,ntimepts)
        data_c, data_t = data_c[:,keep_reps], data_t[:,keep_reps]
        # get stats of background subtracted data

    mean_bs, stdev_bs = get_rep_stats(np.maximum(data_t-data_c,0),nreps,ntimepts)

    # Filter and start at correct initial condition
    if Filter:
        data_c = data_c[keepers,2*nreps:] # keeping keepers and the first two timepoints are before malathion was input
        data_t = data_t[keepers,2*nreps:]
        mean_c, stdev_c = mean_c[keepers,:], stdev_c[keepers,:]
        mean_t, stdev_t = mean_t[keepers,:], stdev_t[keepers,:]
        mean_bs, stdev_bs = mean_bs[keepers,:], stdev_bs[keepers,:]
        geneIDs = [geneIDs[i] for i in keepers]

    # get global snapshot matrices
    Xc = get_global_from_group(data_c,nreps,ntimeptsModel) # for building snapshot matrices
    Xt = get_global_from_group(data_t,nreps,ntimeptsModel)

    # subtract the control from the malathion condition
    X = np.maximum(Xt - Xc,0)

    # normalize such that every gene has zero mean and unit variance over its time series
    if norm:
        scaler = preprocessing.StandardScaler().fit(X.T)
        X = (scaler.transform(X.T)).T

    # print(str([reps[i] for i in range(len(reps))]))
    filenamestr = 'globalsnaps_tps_geneIDs_reps'
    for i in reps:
        filenamestr += str(i)
    if log:
        filenamestr += '_log'
    if norm: 
        filenamestr += '_norm'      

    if save_data:
        pickle.dump([X,ntimeptsModel,geneIDs,reps], open('data/'+filenamestr+'.pickle', 'wb'))

    print(time.time() - start_time, 'seconds')

    return X,geneIDs,Xc,Xt,mean_c,stdev_c,mean_t,stdev_t,mean_bs,stdev_bs






    
