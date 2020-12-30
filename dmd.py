import numpy as np
import time

def get_snapshots_from_global(X,nT,nTraj): 
    '''This function assumes the global snapshot matrix is constructed with trajectories 
        sequentially placed in the columns i.e. 
        t1traj1, t2traj1, t3traj1, ..., tMtraj1, t1traj2, t2traj2, ..., tMtrajJ 
        The output matrices are in the form Xp = [t1traj1, t2traj1, ..., tM-1trajJ]
        Xf = [t2traj1, t3traj1, ..., tMtrajJ]'''
    # nT is number of timepts per trajectory
    # nTraj is the number of trajectories
    
    prevInds = [x for x in range(0,nT-1)]
    forInds = [x for x in range(1,nT)]
    for i in range(0,nTraj-1):
        if i == 0:
            more_prevInds = [x + nT for x in prevInds]
            more_forInds = [x + nT for x in forInds]
        else: 
            more_prevInds = [x + nT for x in more_prevInds]
            more_forInds = [x + nT for x in more_forInds]
        prevInds = prevInds + more_prevInds
        forInds = forInds + more_forInds
    Xp = X[:,prevInds]
    Xf = X[:,forInds]
    return Xp,Xf

def n_step_prediction(A,X,ntimepts,nreps):
    start_time = time.time()
    print('---------Computing MSE for n-step prediction---------')
    X_pred = np.zeros((A.shape[0],ntimepts*nreps))
    count = 0
    for i in range(0,nreps):
        x_test_ic = X[:,i*(ntimepts):i*(ntimepts)+1]
        for j in range(0,ntimepts):
            X_pred[:,count:count+1] = np.dot(np.linalg.matrix_power(A,j),x_test_ic) 
            count += 1
    print(time.time() - start_time, "seconds", "for n-step prediction")

    mse_pred = np.linalg.norm(X - X_pred,2)/(ntimepts-nreps) # minus nreps because initial conditions are given 
    print(f'MSE for n-step prediction is {mse_pred:.3e}')
    return X_pred

def extrapolate(A,X,extrap_horizon,ntimepts,nreps):
    print('---------Extrapolating using DMD model---------')
    start_time = time.time()
    X_extrap = np.zeros((A.shape[0],extrap_horizon*nreps)) 
    count = 0
    for i in range(0,nreps):
        x_test_ic = X[:,i*(ntimepts):i*(ntimepts)+1]
        for j in range(0,extrap_horizon):
            X_extrap[:,count:count+1] = np.dot(np.linalg.matrix_power(A,j),x_test_ic) 
            count += 1
    print(time.time() - start_time, "seconds for extrapolation")
    return X_extrap

def sparsity(A,thresh):
    start_time = time.time()
    print('---------Forcing sparsity in model---------')
    return (np.absolute(A) > thresh) * A

def dmd(X,ntimepts,nreps,extrap_horizon=20,sparse_thresh=2e-3,rank_reduce=False,makeSparse=False,extrapolate=False):
    start_time = time.time()
    print('---------Computing DMD operator---------')
    Xp,Xf = get_snapshots_from_global(X,ntimepts,nreps)
    if rank_reduce == False:
        A = Xf @ np.linalg.pinv(Xp)
    else: 
        U,s,Vh = np.linalg.svd(Xp)
        r = np.minimum(U.shape[1],Vh.shape[0])
        U_r = U[:,0:r] # truncate to rank-r
        s_r = s[0:r]
        Vh_r = Vh[0:r,:]
        Atilde = U_r.T @ Xf @ Vh_r.T @ np.diag(1/s_r) # low-rank dynamics
        A = U_r@Atilde@U_r.T

    l = np.linalg.eigvals(A)
    if np.absolute(l).max() > 1.0:
        print('Model is unstable with mod of eigenvalue',np.absolute(l).max())
    print(time.time() - start_time, 'seconds for DMD')
    if makeSparse:
        A = sparsity(A,sparse_thresh)
    X_pred = n_step_prediction(A,X,ntimepts,nreps)
    if extrapolate:
        X_extrap = extrapolate(A,X,extrap_horizon,ntimepts,nreps)
    else:
        X_extrap = None
    return A,X_pred,X_extrap


