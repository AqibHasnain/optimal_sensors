
import time
import sys
import numpy as np
import pandas as pd # helpful for importing csv
from fastdtw import fastdtw # dynamic time warping distance
from scipy.spatial.distance import euclidean # needed for fastdtw package
from itertools import combinations # to get combinations of replicates 
from scipy.signal import savgol_filter as savgol # for smoothing of time series
from compress_pickle import dump, load

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

def smooth_time_series(data,window_size=7,polyorder=3):
    '''Using scipy's Savitsky-Golay filter to smooth the data. window_size must be an odd number 
    and greater than polyorder'''
    return savgol(data,window_size,polyorder,axis=1)

def acorr(x):
    '''Autocorrelation of a single signal'''
    x = x - x.mean()
    autocorr = np.correlate(x, x, mode='full')
    autocorr /= autocorr.max()
    autocorr = autocorr[len(x)-1:]
    return autocorr

def autocorrelation(data):
    '''Autocorrelation of all signals in data where data is rank three tensor with data for each replicate in third dimension'''
    acorr_all = np.empty((data.shape))
    for ii in range(data.shape[0]):
        for jj in range(data.shape[2]):
            acorr_all[ii,:,jj] = acorr(data[ii,:,jj])
    return acorr_all

def autocorrelation_filter(acorr_data,thresh=0.1):
    # set autocorrelation to 0.0 if it is less than thresh, otherwise set it to 1.0
    acorr_thresh = (np.abs(acorr_data) > thresh)*acorr_data 
    acorr_thresh[np.abs(acorr_thresh) > 0.0] = 1.0
    # if at least L of the autocorrelations (at lags) in each replicate are greater than thresh, 
    # that gene is labeled as non-noisy. To check this criteria, the sum of each row of acorr_thresh 
    # should be greater than or equal to L and then sum of the resulting row (sum over each replicate) 
    # should then be greater than or equal to 3 where boolean matrices are used each step of the way
    L = 1
    acorr_thresh_sum = np.sum(acorr_thresh,axis=1)  
    acorr_thresh_sum = (acorr_thresh_sum >= L) * acorr_thresh_sum
    acorr_thresh_sum[acorr_thresh_sum > 0.0] = 1.0
    acorr_thresh_sum = np.sum(acorr_thresh_sum,axis=1)
    keepers = list((np.nonzero((acorr_thresh_sum == acorr_data.shape[2]) * acorr_thresh_sum))[0]) # 3 as that is # of trajectories
    keepers.sort()
    return keepers

def cv_filter(data,cv_cutoff=0.148,violationAllowance=6):
    # cv_cutoff of 0.148 results in 500 genes, 0.25 gives 3000 genes
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

def dtw_filter(data,thresh=0.02):

    # thresh = 0.00095 results in 506 genes, 0.0095 - 2200 genes, 0.002 - 920 genes
    '''Dynamic time warping distance between a gene's time series measured over two replicates'''
    keepers = []
    for i in range(0,len(data)):
        distance,path = fastdtw(data[i,:,0],data[i,:,1],dist=euclidean)
        distance /= (np.linalg.norm(data[i,:,0],ord=2)*np.linalg.norm(data[i,:,1],ord=2))
        if distance <= thresh: 
            keepers.append(i)
    keepers.sort()
    return keepers

def repr_filter(data_c,data_t,reps,filterMethod,filterThresh): 
    repList = []
    for repcomb in combinations(reps,2): # get lists of all possible length 2 combinations of the replicates being used
        repList.append(list(repcomb)) # if reps=[0,1,2], repList = [[0,1],[0,2],[1,2]]
    if filterMethod == 'CV': # filter based on coefficient of variation
        keepers_c,keepers_t = cv_filter(data_c,cv_cutoff=filterThresh),cv_filter(data_t,cv_cutoff=filterThresh) 
        keepers_repr = list(set(keepers_c)&set(keepers_t))
    elif filterMethod == 'DTW': # filter based on dynamic time warping distance. done in pairs of replicates
        keepers_r = []
        if len(reps) == 3: 
            for these_reps in repList:
                keepers_r.append(dtw_filter(data_c[:,:,these_reps],thresh=filterThresh))
                keepers_r.append(dtw_filter(data_t[:,:,these_reps],thresh=filterThresh))
            keepers_r = [set(k) for k in keepers_r]
            keepers_repr = list(set.intersection(*keepers_r))
        elif len(reps) == 2: 
            keepers_r.append(dtw_filter(data_c),thresh=filterThresh)
            keepers_r.append(dtw_filter(data_t),thresh=filterThresh)
            keepers_r = [set(k) for k in keepers_r]
            keepers_repr = list(set.intersection(*keepers_r))
    print(len(keepers_repr),'genes out of',len(data_c),'passed the reproducibility test: '+filterMethod,'with threshold',filterThresh)
    return keepers_repr

def preprocess(datadir,reps,ntimepts,noiseFilter=False,noiseFilterThresh=0.1,reprFilter=False,reprFilterMethod='CV',\
                                        reprFilterThresh=0.25,Smooth=False,window_size=3,polyorder=1):
    
    '''This function loads the gene expression time series, applies an autocorrelation based filter and either a SNR or DTW based filter to 
    remove noisy and non-reproduceable genes, respectively. This is followed by a smoothing of the data with Savitsky-Golay filters. Finally, 
    the control data is subtracted from the treatment data to produce the differential gene expression dynamics. '''
    print('-'*20,'Preprocessing replicates',reps,'-'*20)

    ''' The df that is loaded has the following structure each column is a datapoint with the progression 
    t1rep1, t1rep2, t1rep3, t1Mrep1, t1Mrep2, t1Mrep3, t2rep1, ..., t12Mrep3 '''

    # load the csv into a pandas dataframe
    df = pd.read_csv(datadir)
    nreps = 3 # number of replicates in the P. fluorescens malathion dataset
    # the column headers (excluding the first) contain the sample labels e.g. time0_condition0_...
    sampleLabels = list(df.columns[1:])
    # the first column contains the transcriptIDs
    transcriptIDs = list(df.iloc[:,0])
    # the rest of the columns contain the time series of gene expression
    df = df.iloc[:,1:] 

    # df contains both conditions. the following splits into dataframes for control and treatment
    data_c, data_t = get_groups_from_df(np.array(df),sampleLabels) 
    # the following reshapes the data appropriately so that they each are rank three tensors with shape nxmxr where n is state dimension, m is number of timepoints, r is number of replicates
    data_c, data_t = put_groups_in_3D(data_c,nreps,ntimepts), put_groups_in_3D(data_t,nreps,ntimepts) 
    # downselecting timepoints: first two timepoints are pre-treatment, last timepoint is anomalous, downselecting replicates: if desired
    data_c, data_t = data_c[:,2:-1,reps], data_t[:,2:-1,reps] 

    if noiseFilter:
        '''
        Criteria to determine if gene has noisy time series: Autocorrelation at first few lags is very small
        Estimation of autocorrelation decreases in accuracy as lag increases, so we consider first few lags only
        Should not use autocorrelation of smoothed data because any noise passed through a smoothing filter will be given a relationship induced by the filter
        '''
        # computing autocorrelation of each time series in the control and treatment data
        acorr_c,acorr_t = autocorrelation(data_c),autocorrelation(data_t)
        # grab just the first three lags excluding the zero lag of course. 
        acorr_c,acorr_t = acorr_c[:,1:4,:], acorr_t[:,1:4,:]
        keepers_noise_c,keepers_noise_t = autocorrelation_filter(acorr_c,thresh=noiseFilterThresh), autocorrelation_filter(acorr_t,thresh=noiseFilterThresh)
        keepers_noise = list(set(keepers_noise_c) & set(keepers_noise_t))
        keepers_noise.sort()
        print(len(keepers_noise),'genes out of',len(df),'passed the noise test: Autocorrelation with at least one of first three lags >',noiseFilterThresh)

    if reprFilter:
        '''
        Though the J experiments (replicates) can be thought of as J realizations from the same dynamical system starting from different initial conditions, it might be useful from a sensor placement 
        standpoint to filter out genes which have time series that are very inconsistent across replicates. To do this, a statistic (coeff. of var.) or metric (distance in pseudotime) is 
        computed.
        '''
        keepers_repr = repr_filter(data_c,data_t,reps,reprFilterMethod,reprFilterThresh)

    # get list of indices corresponding to genes that will be kept for analysis, modeling, and optimization - keepers
    if noiseFilter and reprFilter:
        keepers = list(set(keepers_repr) & set(keepers_noise))
        print(len(keepers),'genes out of',len(df),'are being kept')
        keepers.sort()
    elif noiseFilter and not reprFilter: 
        keepers = keepers_noise
    elif reprFilter and not noiseFilter: 
        keepers = keepers_repr
    elif not reprFilter and not noiseFilter: 
        print('Noise test and reproducibility test were not run. Keeping all genes')
        keepers = list(range(len(df)))

    # downselect the data to keep the genes which passed both tests
    data_c, data_t = data_c[keepers], data_t[keepers]
    # grab the corresponding transcriptIDs
    keep_transcriptIDs = [transcriptIDs[i] for i in keepers]

    if Smooth: 
        ''' 
        One approach to remove noise from a time-series is to use a smoothing spline. Here we apply a Savitsky-Golay filter to each gene's time series
        '''
        data_c = smooth_time_series(data_c,window_size=window_size,polyorder=polyorder)
        data_t = smooth_time_series(data_t,window_size=window_size,polyorder=polyorder)
        print('Smoothing time series of each gene using Savitsky-Golay filter with window_length of',window_size,'and polyorder of',polyorder)

    # Background subtracting the control from the treatment
    X = np.maximum(data_t - data_c,0.0)

    return X,transcriptIDs,keep_transcriptIDs


datadir = 'data/tpm_removed_low_count_genes.csv'
ntimepts = 12 # ntimepts per trajectory (replicate), 12 for monoculture experiment
reps = [0,1,2] # replicates to use. if all J replicates then it would be [0,1,2,...,J-1]
doSave = True # after running the script to downselect genes, do you want to save the downselected df to a CSV? 
if doSave:
    savedir = sys.argv[1] # argument given from command line
doNoiseFilter = True # remove genes which have time series that are classified as noise using its autocorrelation
noiseFilterThresh = 0.1 
doReprFilter = True # remove genes which have time series that are not consistent across replicates
reprFilterMethod = 'DTW' # 'CV' or 'DTW' are the filters that can be applied to identify genes that are reproduceable across replicates
reprFilterThresh = 0.1
doSmooth = True # smooth data after filtering using Savitsky-Golay Filter
window_size, polyorder = 7, 3 # parameters for Savitsky-Golay

X,transcriptIDs,keep_transcriptIDs = preprocess(datadir,reps,ntimepts,noiseFilter=doNoiseFilter,noiseFilterThresh=noiseFilterThresh,reprFilter=doReprFilter,\
                                        reprFilterMethod=reprFilterMethod,reprFilterThresh=reprFilterThresh,Smooth=doSmooth,window_size=window_size,polyorder=polyorder)

if doSave:
    fn = savedir+'/BGSdata_transcriptIDs_keep_transcriptIDs.gz'
    savelist = [X,transcriptIDs,keep_transcriptIDs]
    dump(savelist, fn)



















    
