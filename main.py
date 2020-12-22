
from get_reps_preprocess import *
from dmd import *
from observability import *
from deep_KO_learning import *
import pickle
import sys

datadir = 'data/tpm.csv'
do_deepDMD = False
dumpall = True
ntimepts = 12 # ntimepts per trajectory (replicate)

reps = sys.argv[1]
reps = list(reps.strip('[]').split(','))
reps = [int(i) for i in reps]
# reps = [1] #, [1], [2], [0,1], [0,2], [1,2], [0,1,2] # change this to correspond to the replicates you want to model and analyze
# Note that filtering of genes is always done with all three replicates. 

X,ntimeptsModel,geneIDs,Xc,Xt,mean_c,stdev_c,mean_t,stdev_t,mean_bs,stdev_bs = \
    preprocess(datadir,reps,ntimepts,MEANCUTOFF=0.12,CVCUTOFF=0.1,log=True,norm=False,save_data=False)
# X is the global snapshot matrix after being filtered, and possibly log transformed and possibly normalized, then background subtracted
# ntimeptsNew is the number of timepoints used for training the following model
# geneIDs are the corresponding geneIDs

if not do_deepDMD:
    A,Xpred,Xextrap = dmd(X,ntimeptsModel,len(reps),extrapolate=False)
    # A is the DMD operator
    # X_pred is the matrix of predictions given by A for X
else: 
    ### Neural network parameters ###
    NUM_INPUTS = X.shape[0] # dimension of input
    NUM_HL = 3 # number of hidden layers (excludes the input and output layers)
    NODES_HL = 3 # number of nodes per hidden layer (number of learned observables)
    HL_SIZES = [NODES_HL for i in range(0,NUM_HL+1)] 
    NUM_OUTPUTS = NUM_INPUTS + HL_SIZES[-1] + 1 # output layer takes in dimension of input + 1 + dimension of hl's
    BATCH_SIZE = 1 #int(nT/10) 
    LEARNING_RATE = 0.5
    L2_REG = 0.0
    maxEpochs = 10
    netParams = [NUM_INPUTS,NUM_HL,NODES_HL,HL_SIZES,NUM_OUTPUTS,BATCH_SIZE,LEARNING_RATE,L2_REG,maxEpochs]
    net_name = '/malathion_fluorescens'
    A,W,Xpred,Xextrap = trainKO(Net,netParams,X,ntimeptsModel,len(reps),net_name,save_network=True,extrapolate=False)
    # now also need to add some geneIDs for the extra observables
    for i in range(NODES_HL):
        geneIDs.append('OBSERVABLE'+str(i))

noutputs = 1
nobs_genes = 100
Tf = 9
Wh,idx_maxEnergy,maxEnergy_geneIDs = energy_maximization_single_output(X,A,ntimeptsModel,reps,Tf,geneIDs,nobs_genes)

# Wh,Wh_maxcolNorms,idx_maxcolNorms,obs_geneIDs = observability_maximization_using_model(A,W,noutputs,nobs_genes,geneIDs)
# # Wh is the sampling matrix which we use to maximize observability or output energy
# # Wh_maxcolNorms are the nobs_genes maximum column 1-norms of Wh. 
# # idx_maxcolNorms are the indices corresponding to Wh_maxcolNorms
# # obs_geneIDs are the geneIDs corresponding to Wh_maxcolNorms

maxobs_diffdata,idx_maxobs_diffdata,obs_diffdata_geneIDs = measure_observability_from_diffdata(X,geneIDs,nobs_genes)
# maxobs_diffdata are the nobs_genes maximum observability measures in the differential data
# idx_maxobs_diffdata are the indices corresponding to maxobs_diffdata
# obs_diffdata_geneIDs are the geneIDs corresponding to maxobs_diffdata

maxobs_data,idx_maxobs_data,obs_data_geneIDs = measure_observability_from_data(Xc,Xt,geneIDs,len(reps),ntimeptsModel,nobs_genes)
# maxobs_data are the nobs_genes maximum observability measures in the data
# idx_maxobs_data are the indices corresponding to maxobs_data
# obs_data_geneIDs are the geneIDs corresponding to maxobs_data

if dumpall:
    datadict = {'reps':reps,\
            'ntimepts':ntimepts,\
            'X':X,\
            'ntimeptsModel':ntimeptsModel,\
            'geneIDs':geneIDs,\
            'Xc':Xc,\
            'Xt':Xt,\
            'mean_c':mean_c,\
            'stdev_c':stdev_c,\
            'mean_t':mean_t,\
            'stdev_t':stdev_t,\
            'mean_bs':mean_bs,\
            'stdev_bs':stdev_bs,\
            'A':A,\
            'Xpred':Xpred,\
            'Xextrap':Xextrap,\
            'noutputs':noutputs,\
            'nobs_genes':nobs_genes,\
            'Wh':Wh,\
            'idx_maxEnergy':idx_maxEnergy,\
            'maxEnergy_geneIDs':maxEnergy_geneIDs,\
            'maxobs_data':maxobs_data,\
            'idx_maxobs_data':idx_maxobs_data,\
            'obs_data_geneIDs':obs_data_geneIDs,\
            'maxobs_diffdata':maxobs_diffdata,\
            'idx_maxobs_diffdata':idx_maxobs_diffdata,\
            'obs_diffdata_geneIDs':obs_diffdata_geneIDs,\
            'idx_maxEnergy':idx_maxEnergy,\
            'maxEnergy_geneIDs':maxEnergy_geneIDs}

    namestr = ''
    for i in range(len(reps)):
        namestr += str(reps[i])
    pickle.dump(datadict, open('dataDump'+namestr+'.pickle', 'wb'))

print('\n')


















