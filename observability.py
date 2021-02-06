import numpy as np
import time
from scipy.optimize import minimize

def oneTraj_obj(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    obj = 0
    for i in range(len(At)):
        obj += -( np.linalg.norm(x@At[i]@x0[0],ord=2) )
    return obj

def twoTraj_obj(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    obj = 0
    for i in range(len(At)):
        obj += -( np.linalg.norm(x@At[i]@x0[0],ord=2) + np.linalg.norm(x@At[i]@x0[1],ord=2) )
    return obj

def threeTraj_obj(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    obj = 0
    for i in range(len(At)):
        obj += -( np.linalg.norm(x@At[i]@x0[0],ord=2) + np.linalg.norm(x@At[i]@x0[1],ord=2) + np.linalg.norm(x@At[i]@x0[2],ord=2) )
    return obj

def energy_maximization_single_output(X,A,ntimepts,repnums,Tf,transcriptIDs,IC=0):    
    print('------Optimizing for Optimal State Observer------')
    start_time = time.time()
    C0 = np.random.uniform(0.0,1.0,size=(1,len(A)))
    At = []
    for i in range(0,Tf+1):
        At.append(np.linalg.matrix_power(A,i))
    x0 = []
    if len(repnums) == 1:
        print('------Using one trajectory------')
        x0.append(X[:,IC:IC+1])
        print('Initial objective: ' + str(oneTraj_obj(C0,At,x0,Tf)))
        # optimize
        solution = minimize(oneTraj_obj,C0,args=(At,x0,Tf),method='SLSQP')
        C = (solution.x).reshape(1,C0.shape[1])
        # show final objective
        print('Final objective: ' + str(oneTraj_obj(C,At,x0,Tf)))
    elif len(repnums) == 2:
        print('------Using two trajectories------')
        for i in range(0,len(repnums)):
            # x0.append(X[:,i*ntimepts:i*ntimepts+1])
            x0.append(X[:,i*ntimepts+IC:i*ntimepts+1+IC])
        print('Initial objective: ' + str(twoTraj_obj(C0,At,x0,Tf)))
        # optimize
        solution = minimize(twoTraj_obj,C0,args=(At,x0,Tf),method='SLSQP')
        C = (solution.x).reshape(1,C0.shape[1])
        # show final objective
        print('Final objective: ' + str(twoTraj_obj(C,At,x0,Tf)))
    else:
        print('------Using three trajectories------')
        for i in range(0,len(repnums)):
            # x0.append(X[:,i*ntimepts:i*ntimepts+1])
            x0.append(X[:,i*ntimepts+IC:i*ntimepts+1+IC])
        print('Initial objective: ' + str(threeTraj_obj(C0,At,x0,Tf)))
        # optimize
        solution = minimize(threeTraj_obj,C0,args=(At,x0,Tf),method='SLSQP')
        C = (solution.x).reshape(1,C0.shape[1])
        # show final objective
        print('Final objective: ' + str(threeTraj_obj(C,At,x0,Tf)))

    C = (C/C.max()).T # normalizing C to be b/w 0 and 1
    # idx_maxEnergy = np.flip(C[:,0].argsort()[-100:]) # get indices of 100 max elements of C
    # maxEnergy_geneIDs = [geneIDs[i] for i in idx_maxEnergy]
    print((time.time() - start_time)/60, 'minutes')
    return C
















    