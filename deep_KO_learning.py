import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

class Net(nn.Module):

    ''' 
    This is a feedforward network taking snapshots of a dynamical system
    as input and spews two outputs. Output 1: KPsiXp, the featurized snapshot 
    data propogated forward by the trainable Koopman operator. 
    Output 2: PsiXf, the image of the featurized snapshot data.
    When the network is trained, the two outputs should be close in any distance
    '''
    
    def __init__(self, input_dim, output_dim, hl_sizes):
        super(Net, self).__init__()
        current_dim = input_dim
        self.linears = nn.ModuleList()
        for hl_dim in hl_sizes:
            self.linears.append(nn.Linear(current_dim, hl_dim))
            current_dim = hl_dim
        self.linears.append(nn.Linear(output_dim, output_dim,bias=False))

    def forward(self, x):
        input_vecs = x
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        y = torch.cat((torch.Tensor(np.ones((x.shape[0],1))),input_vecs,x),dim=1)
        x = self.linears[-1](y)
        return {'KPsiXp':x,'PsiXf':y} 
    
def get_paths():
    script_dir = os.path.dirname('deep_KO_learning.py') # getting relative path
    trained_models_path = os.path.join(script_dir, 'trained_models') # which relative path do you want to see
    data_path = os.path.join(script_dir,'data/')
    figs_path = os.path.join(script_dir,'figures')
    return script_dir, trained_models_path, data_path, figs_path

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

    feature_means = np.mean(X,axis=1).reshape(len(X),1)
    cd = 1 - ((np.linalg.norm(X - X_pred,ord=2)**2)/(np.linalg.norm(X - feature_means,ord=2)**2)) # coeff of determination aka R^2 
    print(f'Coefficient of determination for n-step prediction is {cd:.3e}')
    return X_pred,cd

def trainKO(Net,netParams,X,nT,nTraj,net_name,save_network=False):

    script_dir,trained_models_path,data_path,figs_path = get_paths()

    Xp,Xf = X[:,:-1].reshape(len(X),(nT-1)*nTraj,order='F'), X[:,1:].reshape(len(X),(nT-1)*nTraj,order='F')
    X = X.reshape(len(X),(nT)*nTraj,order='F')
    trainXp = torch.Tensor(Xp.T)
    trainXf = torch.Tensor(Xf.T)
    testX = torch.Tensor(X.T)

    numDatapoints = testX.shape[0] # number of total snapshots

    print('Dimension of the state: ' + str(trainXp.shape[1]));
    print('Number of trajectories: ' + str(nTraj));
    print('Number of total snapshots: ' + str(numDatapoints));

    NUM_INPUTS,NUM_HL,NODES_HL,HL_SIZES,NUM_OUTPUTS,BATCH_SIZE,LEARNING_RATE,L2_REG,maxEpochs = netParams

    net = Net(NUM_INPUTS,NUM_OUTPUTS,HL_SIZES)
    print(net)

    ### Defining the loss function and the optimizer ###
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=LEARNING_RATE,weight_decay=L2_REG)

    ### Training the network ###
    print('---------Starting training of network---------')
    print("Using PyTorch Version %s" %torch.__version__)
    start_time = time.time()
    print_less_often = 10
    epoch_to_save_net = 100
    lr_update = 0.95
    eps = 1e-10
    train_loss = []
    prev_loss = 0
    curr_loss = 1e10
    epoch = 0
    net.train()
    while (epoch <= maxEpochs): 

        if epoch % print_less_often == 0:
            if np.abs(prev_loss - curr_loss) < eps:
                break
            prev_loss = curr_loss

        for i in range(0,trainXp.shape[0],BATCH_SIZE):
            
            Kpsixp = net(trainXp[i:i+BATCH_SIZE])['KPsiXp'] 
            psixf = net(trainXf[i:i+BATCH_SIZE])['PsiXf']
            loss = loss_func(psixf, Kpsixp)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = loss.item()

        if epoch % print_less_often == 0:
          print('['+str(epoch)+']'+' loss = '+str(loss.item()))
          if curr_loss > prev_loss: # update learning rate
                for g in optimizer.param_groups:
                    g['lr'] = LEARNING_RATE * lr_update
                LEARNING_RATE = g['lr']
                print('Updated learning rate: ' + str(LEARNING_RATE))

        train_loss.append(loss.item()) 
        epoch+=1

        if epoch % epoch_to_save_net == 0:
            if save_network:
                print('Saving network at epoch ' + str(epoch))
                pickle.dump([NUM_INPUTS,NUM_OUTPUTS,HL_SIZES],open(trained_models_path+net_name+'_netsize.pickle','wb'))
                torch.save(net.state_dict(), trained_models_path+net_name+'_net.pt') # saving the model state in ordered dict
    
    print('['+str(epoch)+']'+' loss = '+ str(loss.item()))
    ### Done training ### 

    K = net.linears[-1].weight[:].detach().numpy()
    PsiX = (net(testX)['PsiXf']).detach().numpy().T

    L, V = np.linalg.eig(K)
    if np.absolute(L).max() > 1.0:
        print('Model is unstable with mod of eigenvalue',np.absolute(L).max())

    # calculate predictions
    PsiXpred,cd = n_step_prediction(K,PsiX,nT-2,nTraj)

    print((time.time() - start_time)/60, "minutes for deep Koopman operator learning")
    ### Saving network (hyper)parameters ###
    if save_network:
        pickle.dump([NUM_INPUTS,NUM_OUTPUTS,HL_SIZES],open(trained_models_path+net_name+'_netsize.pickle','wb'))
        torch.save(net.state_dict(), trained_models_path+net_name+'_net.pt') # saving the model state in ordered dict

    ### Plotting the training loss ###
    import matplotlib.pyplot as plt;
    plt.rcParams.update({'font.size':14});
    plt.rcParams.update({'figure.autolayout': True})
    plt.semilogy(train_loss,lw=4);
    plt.ylabel('MSE Loss');
    plt.xlabel('Epoch');
    plt.savefig(figs_path+net_name+'_Loss.pdf')

    return K,PsiXpred,cd
