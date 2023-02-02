import numpy as np  
import torch
from util import normalize_adj
import torch.optim as optim
from sklearn.metrics import roc_auc_score
#import matplotlib.pyplot as plt
import random
from model import GCN,NN
import math

class GCNBaseline:
    def __init__(self,print_flag=False,early_stopping=True,seed=55,args=None):
        self.printFlag=print_flag
        self.early_stopping=early_stopping
        flag = torch.cuda.is_available()
        print(flag)
        ngpu= 1
        self.device = torch.device(args.device if (torch.cuda.is_available() and ngpu > 0 and args.cuda) else "cpu")
        self.model_name=args.model
        print(self.device)
        print(self.model_name)
        ### set random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass
    def fit(self,adj_list,features,labels,train_mask,val_mask,test_mask,args):
        weight_decay = args.regularization
        self.sample_size=features.size()[0]
        feature_sizes=features.size()[1]
        device = self.device
        self.dropout=args.drop_out
        self.tensor_x = features.to(device)
        self.tensor_y = labels.to(device)
        tensor_train_mask = torch.from_numpy(train_mask).to(device)
        tensor_test_mask = torch.from_numpy(test_mask).to(device)
        tensor_val_mask=torch.from_numpy(val_mask).to(device)
        # 模型定义：Model, Loss, Optimizer
        if self.model_name=='GCN':
            self.sum_adj=normalize_adj(sum(adj_list))
            indices = torch.from_numpy(np.asarray([self.sum_adj.row, self.sum_adj.col]).astype('int64')).long()
            values = torch.from_numpy(self.sum_adj.data.astype(np.float32))
            self.sum_adj=torch.sparse.FloatTensor(indices, values, (self.sample_size,self.sample_size)).to(device)
            self.model = GCN(input_dim=feature_sizes,hidden_dim=args.hidden_dim,out_dim=2).to(device)
            self.pos_weight=torch.tensor([1.0,args.pos_weight],requires_grad=False).to(device)
            ###train
            train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc=self.train(tensor_train_mask,tensor_val_mask,tensor_test_mask,args)

            return train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc
    
    
    def train(self,tensor_train_mask,tensor_val_mask,tensor_test_mask,args,tolerance=100):
        criterion = torch.nn.CrossEntropyLoss(weight=self.pos_weight)
        optimizer = optim.Adam(self.model.parameters(),lr=args.learning_rate, weight_decay=1e-7)
        loss_history = []
        val_auc_history = []
        train_auc_history=[]
        test_auc_history=[]
        self.model.train()
        train_y = self.tensor_y[tensor_train_mask]
        stop_increasing=0
        max_auc=0
        for epoch in range(args.max_steps):
            self.model.train()
            logits = self.model(self.sum_adj,self.tensor_x)  
            train_mask_logits = logits[tensor_train_mask]  
            ### loss correction
            loss=criterion(train_mask_logits,train_y)
            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()    
            train_acc, _, _ = self.test(tensor_train_mask)     
            val_acc, _, _ = self.test(tensor_val_mask)    
            test_acc, _, _ = self.test(tensor_test_mask)
            loss_history.append(loss.item())
            val_auc_history.append(val_acc.item())
            train_auc_history.append(train_acc.item())
            test_auc_history.append(test_acc.item())

            if  val_acc.item()<max_auc:
                stop_increasing+=1
            else:
                max_auc=val_acc.item()
                stop_increasing=0
            if self.early_stopping:
                if stop_increasing>=tolerance:
                    if self.printFlag:
                        print('meet max tolerance on validation set :epoch {}'.format(epoch))
                    break
            if epoch%100==0 and self.printFlag:
                print("Epoch {:03d}: Loss {:.4f}, TrainAuc {:.4}, ValAuc {:.4f}".format(
                    epoch, loss.item(), train_acc.item(), val_acc.item()))
        best_e=np.argmax(val_auc_history)+1
        best_val_auc=max(val_auc_history)
        return loss_history,train_auc_history, val_auc_history,test_auc_history,best_e,best_val_auc
# 测试函数
    def test(self,mask,metric='auc'):
        model=self.model
        model.eval()
        with torch.no_grad():
            logits= model(self.sum_adj,self.tensor_x)
            test_mask_logits = torch.softmax(logits[mask],dim=1)
            if metric=='acc':
                pred=test_mask_logits[:,1]
                acc=(pred==self.tensor_y[mask])
                acc=sum(acc.cpu().numpy())/sum(mask)
                return float(acc.cpu().numpy()), test_mask_logits.cpu().numpy(), self.tensor_y[mask].cpu().numpy()
            elif metric=='auc':
                predict_y = test_mask_logits[:,1]
                auc = roc_auc_score(self.tensor_y[mask].cpu().numpy(),predict_y.cpu().numpy())
                return auc, test_mask_logits.cpu().numpy(), self.tensor_y[mask].cpu().numpy()

class NNBaseline:
    def __init__(self,print_flag=False,early_stopping=True,seed=55,args=None):
        self.printFlag=print_flag
        self.early_stopping=early_stopping
        flag = torch.cuda.is_available()
        print(flag)
        ngpu= 1
        self.device = torch.device(args.device if (torch.cuda.is_available() and ngpu > 0 and args.cuda) else "cpu")
        self.model_name=args.model
        print(self.device)
        print(self.model_name)
        ### set random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass
    def fit(self,features,labels,train_mask,val_mask,test_mask,args):
        weight_decay = args.regularization
        self.sample_size=features.size()[0]
        feature_sizes=features.size()[1]
        device = self.device
        self.dropout=args.drop_out
        self.tensor_x = features.to(device)
        self.tensor_y = labels.to(device)
        tensor_train_mask = torch.from_numpy(train_mask).to(device)
        tensor_test_mask = torch.from_numpy(test_mask).to(device)
        tensor_val_mask=torch.from_numpy(val_mask).to(device)
        # 模型定义：Model, Loss, Optimizer
        if self.model_name=='NN':
            self.model = NN(input_dim=feature_sizes,hidden_dim=args.hidden_dim,out_dim=2).to(device)
            self.pos_weight=torch.tensor([1.0,args.pos_weight],requires_grad=False).to(device)
            ###train
            train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc=self.train(tensor_train_mask,tensor_val_mask,tensor_test_mask,args)

            return train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc
    
    
    def train(self,tensor_train_mask,tensor_val_mask,tensor_test_mask,args,tolerance=50):
        criterion = torch.nn.CrossEntropyLoss(weight=self.pos_weight)
        optimizer = optim.Adam(self.model.parameters(),lr=args.learning_rate, weight_decay=1e-7)
        loss_history = []
        val_auc_history = []
        train_auc_history=[]
        test_auc_history=[]
        self.model.train()
        train_y = self.tensor_y[tensor_train_mask]
        stop_increasing=0
        max_auc=0
        for epoch in range(args.max_steps):
            self.model.train()
            logits = self.model(self.tensor_x)  
            train_mask_logits = logits[tensor_train_mask]  
            ### loss correction
            loss=criterion(train_mask_logits,train_y)
            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()    
            train_acc, _, _ = self.test(tensor_train_mask)     
            val_acc, _, _ = self.test(tensor_val_mask)    
            test_acc, _, _ = self.test(tensor_test_mask)
            loss_history.append(loss.item())
            val_auc_history.append(val_acc.item())
            train_auc_history.append(train_acc.item())
            test_auc_history.append(test_acc.item())
            if  val_acc.item()<max_auc:
                stop_increasing+=1
            else:
                max_auc=val_acc.item()
                stop_increasing=0
            if self.early_stopping:
                if stop_increasing>=tolerance:
                    if self.printFlag:
                        print('meet max tolerance on validation set :epoch {}'.format(epoch))
                    break
            if epoch%100==0 and self.printFlag:
                print("Epoch {:03d}: Loss {:.4f}, TrainAuc {:.4}, ValAuc {:.4f}".format(
                    epoch, loss.item(), train_acc.item(), val_acc.item()))
        best_e=np.argmax(val_auc_history)+1
        best_val_auc=max(val_auc_history)
        return loss_history,train_auc_history, val_auc_history,test_auc_history,best_e,best_val_auc
# 测试函数
    def test(self,mask,metric='auc'):
        model=self.model
        model.eval()
        with torch.no_grad():
            logits= model(self.tensor_x)
            test_mask_logits = torch.softmax(logits[mask],dim=1)
            if metric=='acc':
                pred=test_mask_logits[:,1]
                acc=(pred==self.tensor_y[mask])
                acc=sum(acc.cpu().numpy())/sum(mask)
                return float(acc.cpu().numpy()), test_mask_logits.cpu().numpy(), self.tensor_y[mask].cpu().numpy()
            elif metric=='auc':
                predict_y = test_mask_logits[:,1]
                auc = roc_auc_score(self.tensor_y[mask].cpu().numpy(),predict_y.cpu().numpy())
                return auc, test_mask_logits.cpu().numpy(), self.tensor_y[mask].cpu().numpy()
