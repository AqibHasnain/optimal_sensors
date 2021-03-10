import numpy as np
import time
from scipy.optimize import minimize

def objective(x,At,x0,T):
    ''' Objective function for energy_maximization_single_output() '''
    obj = 0
    for i in range(len(At)):
        obj += -np.linalg.norm(x@At[i]@x0,ord=2)
    return obj

def energy_maximization_single_output(X,A,ntimepts,repnums,Tf,transcriptIDs,IC=0):    
    print('------Optimizing for Optimal State Observer------')
    start_time = time.time()
    r = np.random.randint(0,100)
    np.random.seed(r)
    C0 = np.random.normal(4000,1000.0,size=(1,len(A))) # still trying out various initializations to see what is optimal check out Xavier-Glorot 
    At = []
    for i in range(0,Tf+1):
        At.append(np.linalg.matrix_power(A,i))
    x0 = X[:,0,:]
    print('Initial objective: ' + str(objective(C0,At,x0,Tf)))
    # optimize
    bnds = tuple([(0.0,None) for i in range(X.shape[0])]) # C should be nonnegative
    solution = minimize(objective,C0,args=(At,x0,Tf),method='SLSQP',bounds=bnds)
    C = (solution.x).reshape(1,C0.shape[1])
    # show final objective
    print('Final objective: ' + str(objective(C,At,x0,Tf)))
    C = (C/C.max()).T # normalizing C to be b/w 0 and 1
    print((time.time() - start_time)/60, 'minutes')
    return C,r
















    