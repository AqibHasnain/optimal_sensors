
from dmd import *
from observability import *
from deep_KO_learning import *
from compress_pickle import dump, load
import sys

# this script should be run by calling ./run.sh
# if the data is already preprocessed, comment out the python3 preprocess.py command in run.sh before calling the script
#-------------------------------------USER INPUT REQUIRED HERE----------------------------------------------------#
# IMPORTANT: the file that is being loaded should contain a list of three things, one: the data, two: the IDs for all genes in 
# dataset prior to preprocessing (if desired), three: the transcriptIDs corresponding to the dataset after preprocessing
# items two and three can be the same if preprocessing was not used
datadir = 'run-outputs/' + sys.argv[3] + '/BGSdata_transcriptIDs_keep_transcriptIDs.gz'
saveResults = True # save pickle file with model, optimization result, and random seed used for optimization
dodeepDMD = False # if True, use deep KO learning (pytorch) for system identification (bases and KO). Recommend setting doNorm=True if doDeepDMD
doSaveNN = False # if True, saves neural net params
doSparse = True # if True, induce sparsity via sparsity threshold on the entries of A
if doSparse:
    sparseThresh = 3e-3 # 2e-3
else:
    sparseThresh = 0.0
doReduce = True # if True, reduce dimension of model to min(m,n) where m is numtimepoints and n is dim of state
# there are some additional, not necessary, inputs scattered below e.g. number of hidden layer in network for deepKOlearning
#-----------------------------------------------------------------------------------------------------------------#

X,transcriptIDs,keep_transcriptIDs,keepers,tps_to_keep = load(datadir)
# X is the rank-three tensor of background subtracted data with trajectories in the third dimension
# transcriptIDs is a list of all transcriptIDs imported from the datadir
# keep_transcriptIDs is a list of the transcriptIDs that are being considered and correspond to rows of X

# get the number of timepoints being considered after preprocessing
newntimepts = X.shape[1]
# get the replicates being considered after preprocessing
reps = list(range(X.shape[2]))
print(datadir)

if not dodeepDMD:
    A,newX,cd = dmd(X,newntimepts,len(reps),rank_reduce=doReduce,sparse_thresh=sparseThresh,makeSparse=doSparse)
    # A is the DMD operator
    # newX is subsetted and reshaped (2d) global snapshot matrix
    # X_pred is the matrix of predictions given by A for X
    # cd is coefficient of determination aka R^2 of n.step prediction
else: 
    ### Neural network parameters ###
    NUM_INPUTS = X.shape[0] # dimension of input
    NUM_HL = 3 # number of hidden layers (excludes the input and output layers)
    NODES_HL = 5 # number of nodes per hidden layer (number of learned observables)
    HL_SIZES = [NODES_HL for i in range(0,NUM_HL+1)] 
    NUM_OUTPUTS = NUM_INPUTS + HL_SIZES[-1] + 1 # output layer takes in dimension of input + 1 + dimension of hl's
    BATCH_SIZE = 3 #int(nT/10) 
    LEARNING_RATE = 0.5 # initial learning rate
    L2_REG = 0.0
    maxEpochs = 100
    netParams = [NUM_INPUTS,NUM_HL,NODES_HL,HL_SIZES,NUM_OUTPUTS,BATCH_SIZE,LEARNING_RATE,L2_REG,maxEpochs]
    net_name = '/sampleNet'
    A,newX,Xpred = trainKO(Net,netParams,X,newntimepts,len(reps),net_name,save_network=doSaveNN)
    # now also need to add some transcriptIDs for the extra observables
    for i in range(NODES_HL):
        keep_transcriptIDs.append('OBSERVABLE'+str(i))

if len(sys.argv) <= 4 :
    Tf = newntimepts-1 # finite-horizon for optimization, if not specifying horizon from command line
    ic = 0 # what initial condition should the optimization start with? default is t=0, the actual IC. 
else:
    Tf = int(sys.argv[4])
    ic = int(sys.argv[5])
C,seed = energy_maximization_single_output(X,A,newntimepts,reps,Tf,keep_transcriptIDs,IC=ic) # if not specified, IC=0

if saveResults:
    datadict = {'X':X,'transcriptIDs':transcriptIDs,'keep_transcriptIDs':keep_transcriptIDs,'keepers':keepers,\
                    'tps_kept':tps_to_keep,'A':A,'C':C,'seed':seed}

    namestr = ''
    for i in range(len(reps)):
        namestr += str(reps[i])
    if doSparse:
        namestr += '_sparse'+str(sparseThresh)
    if doReduce:
        namestr += '_reduced'
    namestr += '_IC'+str(ic)+'_m'+str(Tf)
    if len(sys.argv) >= 3:
        namestr += '_'+str(sys.argv[2])
    savedir = 'run-outputs'
    if len(sys.argv) >= 4:
    	savedir += '/'+str(sys.argv[3])
    fn = savedir+'/dump'+namestr+'.gz'
    dump(datadict, fn)

print(fn)
print('\n')

