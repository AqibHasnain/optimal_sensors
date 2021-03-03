
from preprocess import *
from dmd import *
from observability import *
from deep_KO_learning import *
import pickle
import sys


#-------------------------------------USER INPUT REQUIRED HERE----------------------------------------------------#
datadir = 'data/tpm_no_ribo.csv'
saveResults = True # save pickle file with data (filtered&sliced) & model output 
dodeepDMD = False # if True, use deep KO learning (pytorch) for system identification (bases and KO). Recommend setting doNorm=True if doDeepDMD
doSaveNN = False # if True, saves neural net params
doSparse = True # if True, induce sparsity via sparsity threshold on the entries of A
if doSparse:
    sparseThresh = 3e-3 # 2e-3
else:
    sparseThresh = 0.0
doReduce = True # if True, reduce dimension of model to min(m,n) where m is numtimepoints and n is dim of state
ntimepts = 12 # ntimepts per trajectory (replicate), 12 for monoculture experiment (tpm.csv)
doFilter = True # set this to False if you don't want to remove any genes from the analysis
filter_method = 'DTW' # 'CV', 'MD', or 'DTW'
doFilterB4BackSub = True
doNorm = False # set this to True to normalize data before filtering
if len(sys.argv) < 2:
    reps = [0,1,2] # if not specifying reps from command line, put reps to use in list here
else:
    reps = sys.argv[1]
    reps = list(reps.strip('[]').split(','))
    reps = [int(i) for i in reps] 
# there are some additional, not necessary, inputs scattered below e.g. number of hidden layer in network for deepKOlearning
#-----------------------------------------------------------------------------------------------------------------#

X,transcriptIDs,keepers,newntimepts = preprocess(datadir,reps,ntimepts,Norm=doNorm,Filter=doFilter,filterMethod=filter_method,\
                                                filterB4BackSub=doFilterB4BackSub)

if not dodeepDMD:
    A,percent_nonzero_to_zero,newX,cd = dmd(X,newntimepts,len(reps),rank_reduce=doReduce,sparse_thresh=sparseThresh,makeSparse=doSparse)
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
        transcriptIDs.append('OBSERVABLE'+str(i))

if len(sys.argv) <= 4 :
    Tf = newntimepts-1 # finite-horizon for optimization, if not specifying horizon from command line
    ic = 0 # what initial condition should the optimization start with? default is t=0, the actual IC. 
else:
    Tf = int(sys.argv[4])
    ic = int(sys.argv[5])

keep_transcriptIDs = [transcriptIDs[i] for i in keepers]

C,C0 = energy_maximization_single_output(newX,A,newntimepts,reps,Tf,keep_transcriptIDs,IC=ic) # if not specified, IC=0

if saveResults:
    datadict = {'X':X,\
            'transcriptIDs':transcriptIDs,\
            'keepers':keepers,\
            'A':A,\
            'C':C,\
            'C0':C0}

    namestr = ''
    for i in range(len(reps)):
        namestr += str(reps[i])
    if doNorm:
        namestr += '_norm'
    if doSparse:
        namestr += '_sparse'+str(sparseThresh)
    if doReduce:
        namestr += '_reduced'
    if doFilterB4BackSub:
        namestr += '_filterB4BS_'+str(filter_method)
    else:
        namestr += '_filterAfterBS_'+str(filter_method)
    namestr += '_IC'+str(ic)+'_m'+str(Tf)

    if len(sys.argv) >= 3:
        namestr += '_'+str(sys.argv[2])

    savedir = 'run-outputs'
    if len(sys.argv) >= 4:
    	savedir += '/'+str(sys.argv[3])

    fn = savedir+'/dump'+namestr+'.pickle'
    pickle.dump(datadict, open(fn, 'wb'))

print(fn)
print('\n')


















