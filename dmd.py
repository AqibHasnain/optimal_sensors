import numpy as np
import time

def n_step_prediction(A,X,ntimepts,nreps):
    start_time = time.time()
    print('---------Computing R^2 for n-step prediction---------')
    X_pred = np.zeros((A.shape[0],ntimepts*nreps))
    count = 0
    for i in range(0,nreps):
        x_test_ic = X[:,i*(ntimepts):i*(ntimepts)+1]
        for j in range(0,ntimepts):
            X_pred[:,count:count+1] = np.dot(np.linalg.matrix_power(A,j),x_test_ic) 
            count += 1
    print(time.time() - start_time, "seconds", "for n-step prediction")

    feature_means = np.mean(X,axis=1).reshape(len(X),1)
    cd = 1 - ((np.linalg.norm(X - X_pred,ord=2)**2)/(np.linalg.norm(X - feature_means,ord=2)**2))   # coeff of determination aka R^2 
    print(f'Coefficient of determination for n-step prediction is {cd:.3e}')
    return X_pred, cd

def sparsity(A,thresh):
    start_time = time.time()
    print('---------Forcing sparsity in model with threshhold',thresh,'---------')
    return (np.absolute(A) > thresh) * A

def dmd(X,ntimepts,nreps,sparse_thresh=2e-3,rank_reduce=False,makeSparse=False,extrapolate=False):
    start_time = time.time()
    print('---------Computing DMD operator---------')
    Xp,Xf = X[:,:-1].reshape(len(X),(ntimepts-1)*nreps,order='F'), X[:,1:].reshape(len(X),(ntimepts-1)*nreps,order='F')
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

    # Check if model is stable
    l = np.linalg.eigvals(A)
    # if np.absolute(l).max() > 1.0:
        # print('Model is unstable with mod of eigenvalue',np.absolute(l).max())
    print((time.time() - start_time)/60, 'minutes for DMD')

    # make model sparse 
    if makeSparse:
        A = sparsity(A,sparse_thresh)

    # calculate prediction accuracy 
    X = X.reshape(len(X),(ntimepts)*nreps,order='F')
    X_pred, cd = n_step_prediction(A,X,ntimepts,nreps)

    return A,X,X_pred,cd

