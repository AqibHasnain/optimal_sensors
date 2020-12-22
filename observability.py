import numpy as np
import time
from scipy.optimize import minimize

def observability_maximization_using_model(A,W,noutputs,nobs_genes,geneIDs):
    print('---------Observability analysis of model---------')
    start_time = time.time()
    nobs = A.shape[0]
    Wh = np.dot(np.concatenate((np.identity(noutputs),np.zeros((noutputs,nobs-noutputs))),axis=1),W)
    colNorms = []
    for i in range(0,Wh.shape[1]):
        colNorms.append(np.linalg.norm(Wh[:,i],ord=1))
    colNorms = np.array(colNorms)
    idx_maxcolNorms = np.flip(colNorms.argsort()[-nobs_genes:])
    maxcolNorms = colNorms[np.flip(colNorms.argsort()[-nobs_genes:])]
    obs_geneIDs = [geneIDs[i] for i in idx_maxcolNorms]
    print(time.time() - start_time, 'seconds')
    return Wh,maxcolNorms,idx_maxcolNorms,obs_geneIDs

def measure_observability_from_diffdata(X,geneIDs,nobs_genes):
    print('---------Calculating observability from differential data---------')
    start_time = time.time()
    obs_data = np.linalg.norm(X,ord=2,axis=1)
    idx_maxobs_data = np.flip(obs_data.argsort()[-nobs_genes:])
    maxobs_data = obs_data[np.flip(obs_data.argsort()[-nobs_genes:])]
    obs_data_geneIDs = [geneIDs[i] for i in idx_maxobs_data]
    print(time.time() - start_time, 'seconds')
    return maxobs_data,idx_maxobs_data,obs_data_geneIDs

def measure_observability_from_data(Xc,Xt,geneIDs,nreps,ntimepts,nobs_genes):
    # ||x(all_time)_treatment||/||x(0) malathion|| - ||x(all_time)_control||/||x(0)_control|| 
    # taking mean over each replicate to get final observability measure
    print('---------Calculating observability from control and treatment data---------') 
    start_time = time.time()
    obs_c = np.zeros([Xc.shape[0],nreps])
    obs_t = np.zeros([Xt.shape[0],nreps])
    for i in range(0,nreps):
        obs_c[:,i] = np.linalg.norm(Xc[:,i*ntimepts:i*ntimepts+ntimepts],ord=2,axis=1)#/np.linalg.norm(Xc[:,i*ntimepts],ord=2)
        obs_t[:,i] = np.linalg.norm(Xt[:,i*ntimepts:i*ntimepts+ntimepts],ord=2,axis=1)#/np.linalg.norm(Xt[:,i*ntimepts],ord=2)
    obs_data = np.mean(np.maximum(obs_t - obs_c,0),axis=1)
    idx_maxobs_data = np.flip(obs_data.argsort()[-nobs_genes:])
    maxobs_data = obs_data[np.flip(obs_data.argsort()[-nobs_genes:])]
    obs_data_geneIDs = [geneIDs[i] for i in idx_maxobs_data]
    print(time.time() - start_time, 'seconds')
    return maxobs_data,idx_maxobs_data,obs_data_geneIDs

def oneTraj_obj(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    return - ( np.linalg.norm(x@At[0]@x0[0],ord=2) + np.linalg.norm(x@At[1]@x0[0],ord=2) + \
             np.linalg.norm(x@At[2]@x0[0],ord=2) + np.linalg.norm(x@At[3]@x0[0],ord=2) + \
             np.linalg.norm(x@At[4]@x0[0],ord=2) + np.linalg.norm(x@At[5]@x0[0],ord=2) + \
             np.linalg.norm(x@At[6]@x0[0],ord=2) + np.linalg.norm(x@At[7]@x0[0],ord=2) + \
             np.linalg.norm(x@At[8]@x0[0],ord=2) + np.linalg.norm(x@At[9]@x0[0],ord=2) )

def twoTraj_obj(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    return - ( np.linalg.norm(x@At[0]@x0[0],ord=2) + np.linalg.norm(x@At[1]@x0[0],ord=2) + \
             np.linalg.norm(x@At[2]@x0[0],ord=2) + np.linalg.norm(x@At[3]@x0[0],ord=2) + \
             np.linalg.norm(x@At[4]@x0[0],ord=2) + np.linalg.norm(x@At[5]@x0[0],ord=2) + \
             np.linalg.norm(x@At[6]@x0[0],ord=2) + np.linalg.norm(x@At[7]@x0[0],ord=2) + \
             np.linalg.norm(x@At[8]@x0[0],ord=2) + np.linalg.norm(x@At[9]@x0[0],ord=2) + \
             np.linalg.norm(x@At[0]@x0[1],ord=2) + np.linalg.norm(x@At[1]@x0[1],ord=2) + \
             np.linalg.norm(x@At[2]@x0[1],ord=2) + np.linalg.norm(x@At[3]@x0[1],ord=2) + \
             np.linalg.norm(x@At[4]@x0[1],ord=2) + np.linalg.norm(x@At[5]@x0[1],ord=2) + \
             np.linalg.norm(x@At[6]@x0[1],ord=2) + np.linalg.norm(x@At[7]@x0[1],ord=2) + \
             np.linalg.norm(x@At[8]@x0[1],ord=2) + np.linalg.norm(x@At[9]@x0[1],ord=2) )

def threeTraj_obj(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    return - ( np.linalg.norm(x@At[0]@x0[0],ord=2) + np.linalg.norm(x@At[1]@x0[0],ord=2) + \
             np.linalg.norm(x@At[2]@x0[0],ord=2) + np.linalg.norm(x@At[3]@x0[0],ord=2) + \
             np.linalg.norm(x@At[4]@x0[0],ord=2) + np.linalg.norm(x@At[5]@x0[0],ord=2) + \
             np.linalg.norm(x@At[6]@x0[0],ord=2) + np.linalg.norm(x@At[7]@x0[0],ord=2) + \
             np.linalg.norm(x@At[8]@x0[0],ord=2) + np.linalg.norm(x@At[9]@x0[0],ord=2) + \
             np.linalg.norm(x@At[0]@x0[1],ord=2) + np.linalg.norm(x@At[1]@x0[1],ord=2) + \
             np.linalg.norm(x@At[2]@x0[1],ord=2) + np.linalg.norm(x@At[3]@x0[1],ord=2) + \
             np.linalg.norm(x@At[4]@x0[1],ord=2) + np.linalg.norm(x@At[5]@x0[1],ord=2) + \
             np.linalg.norm(x@At[6]@x0[1],ord=2) + np.linalg.norm(x@At[7]@x0[1],ord=2) + \
             np.linalg.norm(x@At[8]@x0[1],ord=2) + np.linalg.norm(x@At[9]@x0[1],ord=2) + \
             np.linalg.norm(x@At[0]@x0[2],ord=2) + np.linalg.norm(x@At[1]@x0[2],ord=2) + \
             np.linalg.norm(x@At[2]@x0[2],ord=2) + np.linalg.norm(x@At[3]@x0[2],ord=2) + \
             np.linalg.norm(x@At[4]@x0[2],ord=2) + np.linalg.norm(x@At[5]@x0[2],ord=2) + \
             np.linalg.norm(x@At[6]@x0[2],ord=2) + np.linalg.norm(x@At[7]@x0[2],ord=2) + \
             np.linalg.norm(x@At[8]@x0[2],ord=2) + np.linalg.norm(x@At[9]@x0[2],ord=2) )


def energy_maximization_single_output(X,A,ntimepts,repnums,Tf,geneIDs,nobs_genes):    
    print('------Optimizing for Optimal State Observer------')
    start_time = time.time()
    C0 = np.random.uniform(0.0,1.0,size=(1,len(A)))
    At = []
    for i in range(0,Tf+1):
        At.append(np.linalg.matrix_power(A,i))
    x0 = []
    if len(repnums) == 1:
        print('------Using one trajectory------')
        x0.append(X[:,0:1])
        print('Initial objective: ' + str(oneTraj_obj(C0,At,x0,Tf)))
        # optimize
        solution = minimize(oneTraj_obj,C0,args=(At,x0,Tf),method='SLSQP')
        C = (solution.x).reshape(1,C0.shape[1])
        # show final objective
        print('Final objective: ' + str(oneTraj_obj(C,At,x0,Tf)))
    elif len(repnums) == 2:
        print('------Using two trajectories------')
        for i in range(0,len(repnums)):
            x0.append(X[:,i*ntimepts:i*ntimepts+1])
        print('Initial objective: ' + str(twoTraj_obj(C0,At,x0,Tf)))
        # optimize
        solution = minimize(twoTraj_obj,C0,args=(At,x0,Tf),method='SLSQP')
        C = (solution.x).reshape(1,C0.shape[1])
        # show final objective
        print('Final objective: ' + str(twoTraj_obj(C,At,x0,Tf)))
    else:
        print('------Using three trajectories------')
        for i in range(0,len(repnums)):
            x0.append(X[:,i*ntimepts:i*ntimepts+1])
        print('Initial objective: ' + str(threeTraj_obj(C0,At,x0,Tf)))
        # optimize
        solution = minimize(threeTraj_obj,C0,args=(At,x0,Tf),method='SLSQP')
        C = (solution.x).reshape(1,C0.shape[1])
        # show final objective
        print('Final objective: ' + str(threeTraj_obj(C,At,x0,Tf)))

    C = (C/C.max()).T # normalizing C to be b/w 0 and 1
    idx_maxEnergy = np.flip(C[:,0].argsort()[-nobs_genes:])
    maxEnergy_geneIDs = [geneIDs[i] for i in idx_maxEnergy]
    print((time.time() - start_time)/60, 'minutes')
    return C,idx_maxEnergy,maxEnergy_geneIDs
















    