import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter

df = pickle.load( open('dict_full.pickle','rb' ) )

ntimepts = df['ntimepts']
reps = df['reps']
X = df['X']
ntimeptsModel = df['ntimeptsModel']
geneIDs = df['geneIDs']
Xc = df['Xc']
Xt = df['Xt']
mean_c = df['mean_c']
stdev_c = df['stdev_c']
mean_t = df['mean_t']
stdev_t = df['stdev_t']
mean_bs = df['mean_bs'] 
stdev_bs = df['stdev_bs']
A = df['A']
W = df['W']
Xpred = df['Xpred']
Xextrap = df['Xextrap']
noutputs = df['noutputs']
nobs_genes = df['nobs_genes']
Wh = df['Wh']
Wh_maxcolNorms = df['Wh_maxcolNorms'] # observability measure from old approach using model. 
idx_maxcolNorms = df['idx_maxcolNorms'] 
obs_geneIDs = df['obs_geneIDs']
maxobs_data = df['maxobs_data'] # corresponds to using malathion and control data for observability measure
idx_maxobs_data = df['idx_maxobs_data'] 
obs_data_geneIDs = df['obs_data_geneIDs']
maxobs_diffdata = df['maxobs_diffdata'] # corresponds to using differential data for observability measure
idx_maxobs_diffdata = df['idx_maxobs_diffdata']
obs_diffdata_geneIDs = df['obs_diffdata_geneIDs']

tspan = np.linspace(0,110,ntimepts) # 110 is the last sampling time in our experiment
nreps = len(reps)

def plot_means_opt_sensor_genes(names,idx,mean,stdev,nrows=2,ncols=2,savefigs=False,figname='autosavefig'):
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 20))
    c = 0
    ymin = mean[idx].min()-0.1
    ymax = mean[idx].max()+0.1
    for row in range(0,nrows):
        for col in range(0,ncols):
            ax[row,col].title.set_text(names[c])
            ax[row,col].errorbar(tspan,mean[idx[c],:],\
                        yerr=df['stdev_bs'][idx[c],:],\
                        fmt='D--',c='tab:green',ms=10,ecolor='tab:green',capsize=7);
            ax[row,col].spines['right'].set_visible(False)
            ax[row,col].spines['top'].set_visible(False)
            ax[row,col].set_ylim([ymin,ymax])
            c += 1
    if savefigs:
        plt.savefig('figures/' + figname + '.pdf')
            
def plots_opt_sensor_genes(names,idx,mean_c,stdev_c,mean_bs,stdev_bs,X,Xpred,Xc,Xt,ntimeptsModel,savefigs=False,figname='autosavefig'):
    gene_to_plot = idx[0] #idx[np.random.randint(0,len(idx))]
    print(gene_to_plot, names[gene_to_plot])

    nrows = 1
    ncols = 3
    fig1, ax1 = plt.subplots(nrows, ncols, figsize=(15, 3));
    ax1[0].errorbar(tspan,mean_c[gene_to_plot,:],yerr=stdev_c[gene_to_plot,:],fmt='s--',ms=10,ecolor='tab:blue',capsize=7);
    ax1[0].errorbar(tspan,mean_t[gene_to_plot,:],yerr=stdev_t[gene_to_plot,:],fmt='o--',ms=10,ecolor='tab:orange',capsize=7);
    ax1[1].errorbar(tspan[:],mean_bs[gene_to_plot,:],yerr=stdev_bs[gene_to_plot,:],fmt='d--',c='tab:green',ms=10,ecolor='tab:green',capsize=7);
    for i in range(0,nreps):
        ax1[2].plot(tspan[2:],Xc[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],'s--',ms=10,color='tab:blue');
        ax1[2].plot(tspan[2:],Xt[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],'o--',ms=10,color='tab:orange');    
    for i in range(0,ncols):
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)
    ax1[0].title.set_text('mean of the groups')
    ax1[1].title.set_text('mean of the background subtracted data')
    ax1[2].title.set_text('all trajectories of the groups')
    if savefigs:
        plt.savefig('figures/' + figname + '1.pdf')

    nrows = 1
    ncols = nreps
    fig2, ax2 = plt.subplots(nrows, ncols, figsize=(15, 3));
    ymin = np.minimum(Xc[gene_to_plot].min(),Xt[gene_to_plot].min())-0.1
    ymax = np.maximum(Xc[gene_to_plot].max(),Xt[gene_to_plot].max())+0.1
    for i in range(0,nreps):
        ax2[i].set_title('Replicate'+str(i+1))
        ax2[i].plot(tspan[2:],Xc[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],'s--',ms=10,color='tab:blue');
        ax2[i].plot(tspan[2:],Xt[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],'o--',ms=10,color='tab:orange');    
#         ax[i].plot(tspan[2:],X[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],'D--',ms=10,color='tab:green');    
        ax2[i].set_ylim([ymin,ymax])
    if savefigs:
        plt.savefig('figures/' + figname + '2.pdf')
    
    nrows = 1
    ncols = nreps
    np.linalg
    fig3, ax3 = plt.subplots(nrows, ncols, figsize=(15, 3));
    for i in range(0,nreps):
        ax3[i].plot(savgol_filter(X[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],5,3),'D--',ms=10,color='tab:green');    
        ax3[i].plot(savgol_filter(Xpred[gene_to_plot,i*(ntimeptsModel):i*(ntimeptsModel)+(ntimeptsModel)],5,3),'X--',ms=10,color='tab:red');    
    if savefigs:
        plt.savefig('figures/' + figname +'3.pdf')

NROWS = 5
NCOLS = 5 # there are 25 selected sensor genes. 

DOSAVEFIGS = True

# plot means of background subtracted optimal sensor genes given by observability maximization using model
plot_means_opt_sensor_genes(obs_geneIDs,idx_maxcolNorms,mean_bs,stdev_bs,nrows=NROWS,ncols=NCOLS,savefigs=DOSAVEFIGS,figname='sensorgenes_model')

# various plots for optimal sensor genes given by observability maximization using model
plots_opt_sensor_genes(geneIDs,idx_maxcolNorms,mean_c,stdev_c,mean_bs,stdev_bs,X,Xpred,Xc,Xt,ntimeptsModel,savefigs=DOSAVEFIGS,figname='sensorgenes_model')

# plot means of background subtracted optimal sensor genes given by observability measures from differential data
plot_means_opt_sensor_genes(obs_diffdata_geneIDs,idx_maxobs_diffdata,mean_bs,stdev_bs,nrows=NROWS,ncols=NCOLS,savefigs=DOSAVEFIGS,figname='sensorgenes_diffdata')

# various plots for optimal sensor genes given by observability measure of differential data
plots_opt_sensor_genes(geneIDs,idx_maxobs_diffdata,mean_c,stdev_c,mean_bs,stdev_bs,X,Xpred,Xc,Xt,ntimeptsModel,savefigs=DOSAVEFIGS,figname='sensorgenes_diffdata')

# plot means of background subtracted optimal sensor genes given by observability measures from treatment and control
plot_means_opt_sensor_genes(obs_data_geneIDs,idx_maxobs_data,mean_bs,stdev_bs,nrows=NROWS,ncols=NCOLS,savefigs=DOSAVEFIGS,figname='sensorgenes_data')

# various plots for optimal sensor genes given by observability measure of treatment and control data
plots_opt_sensor_genes(geneIDs,idx_maxobs_data,mean_c,stdev_c,mean_bs,stdev_bs,X,Xpred,Xc,Xt,ntimeptsModel,savefigs=DOSAVEFIGS,figname='sensorgenes_data')

















