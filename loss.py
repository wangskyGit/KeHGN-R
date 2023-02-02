import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions

def sample_sieve():
    """
    dynamic sample sieve
    """
    return



def loss_sieve(epoch, y, t,ind,loss_all,loss_div_all,device, noise_prior = None, pos_weight=None):
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduction = 'none')
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y,dim=1) + 1e-8)
    #loss_ = -torch.log(F.softmax(y[t==0],dim=1) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1) *(t==0)
    neg_loss,neg_indics=loss_sel[t==0].sort(descending=True)
    neg_loss=neg_loss.detach().cpu().numpy()
    t_max=neg_loss[math.floor(noise_prior*(sum(t==1)))]
    # if noise_prior is None:
    #     loss =  loss - beta*torch.mean(loss_,1)
    # else:
    #     loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind.cpu().numpy(),epoch] = loss_numpy
    loss_div_all[ind.cpu().numpy(),epoch] = loss_div_numpy
    
    #t_max=-0.5### threholds for sample sieve
    for i in range(len(loss_numpy)):
        if epoch <=10:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= t_max or t.cpu().numpy()[i]==1:##低于阈值或者标签为1的通过sample sieve
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda(device=device)
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

        

def loss_cores(epoch, y, t,ind,loss_all,loss_div_all,device,noise_prior = None, pos_weight=None):
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduction = 'none',weight=pos_weight)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y,dim=1) + 1e-8)
    #loss_ = -torch.log(F.softmax(y[t==0],dim=1) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1) *(t==0)
    
    neg_loss,neg_indics=loss_sel[t==0].sort(descending=True)
    neg_loss=neg_loss.detach().cpu().numpy()
    t_max=neg_loss[math.floor(noise_prior*(sum(t==1)))]
    #t_max=-0.3  ### threholds for sample sieve
    
    loss =  loss - beta*torch.mean(loss_,1)

    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind.cpu().numpy(),epoch] = loss_numpy
    loss_div_all[ind.cpu().numpy(),epoch] = loss_div_numpy
    for i in range(len(loss_numpy)):
        if epoch <=10:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= t_max or t.cpu().numpy()[i]==1:##低于阈值或者标签为1的通过sample sieve
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda(device=device)
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 0.5, num=90)
    beta3 = np.linspace(0.5, 1.0, num=1000)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]
