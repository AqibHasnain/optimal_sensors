import numpy as np
import time

def observability_maximization_using_model(A,W,noutputs,nobs_genes,geneIDs):
	print('---Observability analysis of model---')
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
	print('---Calculating observability from differential data---')
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
	print('---Caclulating observability from control and treatment data') 
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
	