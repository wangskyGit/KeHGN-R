from distutils.log import error
import numpy as np  
import torch
from util import normalize_adj
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
#import matplotlib.pyplot as plt
import random
from model import KeGCN,KeGN
from loss import loss_cores
from tm import transition_matrix_learn,bltm
import math

class ExpModel:
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
        self.sample_size=features[0].size()[0]
        feature_sizes=[f.size()[1] for f in features]
        device = self.device
        self.dropout=args.drop_out
        self.tensor_x = [f.to(device) for f in features]
        self.tensor_y = labels.to(device)
        tensor_train_mask = torch.from_numpy(train_mask).to(device)
        tensor_test_mask = torch.from_numpy(test_mask).to(device)
        tensor_val_mask=torch.from_numpy(val_mask).to(device)
        # 模型定义：Model, Loss, Optimizer
        
        if not args.nke:
            self.use_kg=True
            self.kg_dim=feature_sizes[1]
        else:
            self.use_kg=False
            self.kg_dim=None
        if self.model_name=='KeGCNR':
            self.tensor_adjacency_list=[]
            self.sum_adj=normalize_adj(sum(adj_list))
            indices = torch.from_numpy(np.asarray([self.sum_adj.row, self.sum_adj.col]).astype('int64')).long()
            values = torch.from_numpy(self.sum_adj.data.astype(np.float32))
            self.sum_adj=torch.sparse.FloatTensor(indices, values, (self.sample_size,self.sample_size)).to(device)
            for adj in adj_list:
                indices = torch.from_numpy(np.asarray([adj.row, adj.col]).astype('int64')).long()
                values = torch.from_numpy(adj.data.astype(np.float32))
                self.tensor_adjacency_list.append(torch.sparse.FloatTensor(indices, values, (self.sample_size,self.sample_size)).to(device))
            if args.att_act=='LeakyRelu':
                att_activate=torch.nn.LeakyReLU()
            elif args.att_act=='Tanh':
                att_activate=torch.nn.Tanh()
            else:
                raise error('unsupport activation function, please try LeakyRelu or Tanh')

            self.model = KeGCN(self.tensor_adjacency_list,input_dim=feature_sizes[0],device=self.device,embed_dim=args.hidden_dim,att_activate=att_activate,
                            use_kg=self.use_kg,kg_dim=self.kg_dim,share=args.share).to(device)
            self.pretrain_model = KeGCN(self.tensor_adjacency_list,input_dim=feature_sizes[0],device=self.device,embed_dim=args.hidden_dim,att_activate=att_activate,
                            use_kg=self.use_kg,kg_dim=self.kg_dim,share=args.share).to(device)
            # self.model = KeGN(self.tensor_adjacency_list,input_dim=feature_sizes[0],device=self.device,embed_dim=args.hidden_dim,att_activate=att_activate,
            #                 use_kg=self.use_kg,kg_dim=self.kg_dim,share=args.share).to(device)
            # self.pretrain_model = KeGN(self.tensor_adjacency_list,input_dim=feature_sizes[0],device=self.device,embed_dim=args.hidden_dim,att_activate=att_activate,
            #                 use_kg=self.use_kg,kg_dim=self.kg_dim,share=args.share).to(device)
            self.pos_weight=torch.tensor([1.0,args.pos_weight],requires_grad=False).to(device)
            
            ###pretrain to get bayes optimal labels
            train_loss,train_auc,loss_v=self.pretrain(tensor_train_mask,tensor_val_mask,args.pretrain_steps,args)
            distilled_label=[]##[[noised],[true]]
            noised_label=[]
            distilled_mask=train_mask+val_mask
            noised_label=labels[train_mask+val_mask]
            for i in range(0,len(loss_v)):
                if loss_v[i]==0:
                    distilled_label.append(1)
                if loss_v[i]==1:
                    distilled_label.append(noised_label[i])
            distilled_label=torch.LongTensor(distilled_label)
            noised_label=torch.LongTensor(noised_label)
            ###trainsition matrix learning
            bayes_tm,tm_model=transition_matrix_learn(distilled_label,noised_label,self.tensor_x[0],distilled_mask,self.sum_adj,args)
            #bayes_tm,tm_model=bltm(distilled_label,noised_label,self.tensor_x[0],distilled_mask,args)
            ###training with loss correction
            train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc=self.train(tensor_train_mask,tensor_val_mask,tensor_test_mask,args,tm_model)
            return train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc
        if self.model_name=='COREAS':
            self.tensor_adjacency_list=[]
            self.sum_adj=normalize_adj(sum(adj_list))
            indices = torch.from_numpy(np.asarray([self.sum_adj.row, self.sum_adj.col]).astype('int64')).long()
            values = torch.from_numpy(self.sum_adj.data.astype(np.float32))
            self.sum_adj=torch.sparse.FloatTensor(indices, values, (self.sample_size,self.sample_size)).to(device)
            for adj in adj_list:
                indices = torch.from_numpy(np.asarray([adj.row, adj.col]).astype('int64')).long()
                values = torch.from_numpy(adj.data.astype(np.float32))
                self.tensor_adjacency_list.append(torch.sparse.FloatTensor(indices, values, (self.sample_size,self.sample_size)).to(device))
            if args.att_act=='LeakyRelu':
                att_activate=torch.nn.LeakyReLU()
            elif args.att_act=='Tanh':
                att_activate=torch.nn.Tanh()
            else:
                raise error('unsupport activation function, please try LeakyRelu or Tanh')
            self.pretrain_model = KeGCN(self.tensor_adjacency_list,input_dim=feature_sizes[0],device=self.device,embed_dim=args.hidden_dim,att_activate=att_activate,
                            use_kg=self.use_kg,kg_dim=self.kg_dim,share=args.share).to(device)
            self.pos_weight=torch.tensor([1.0,args.pos_weight],requires_grad=False).to(device)
            train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc=self.cores_train(tensor_train_mask,tensor_val_mask,tensor_test_mask,args)
            return train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc
        if self.model_name=='KeGCN':
            self.tensor_adjacency_list=[]
            for adj in adj_list:
                indices = torch.from_numpy(np.asarray([adj.row, adj.col]).astype('int64')).long()
                values = torch.from_numpy(adj.data.astype(np.float32))
                self.tensor_adjacency_list.append(torch.sparse.FloatTensor(indices, values, (self.sample_size,self.sample_size)).to(device))
            if args.att_act=='LeakyRelu':
                att_activate=torch.nn.LeakyReLU()
            elif args.att_act=='Tanh':
                att_activate=torch.nn.Tanh()
            else:
                raise error('unsupport activation function, please try LeakyRelu or Tanh')

            self.model = KeGCN(self.tensor_adjacency_list,input_dim=feature_sizes[0],device=self.device,embed_dim=args.hidden_dim,att_activate=att_activate,
                            use_kg=self.use_kg,kg_dim=self.kg_dim,share=args.share).to(device)
            self.pos_weight=torch.tensor([1.0,args.pos_weight],requires_grad=False).to(device)
            weight=[1.0,args.pos_weight]
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,requires_grad=False).to(device))
            self.optimizer = optim.Adam(self.model.parameters(),lr=args.learning_rate,weight_decay=weight_decay)
            
            ###train
            train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc=self.train(tensor_train_mask,tensor_val_mask,tensor_test_mask,args)

            return train_loss,train_auc,val_auc,test_auc,best_e,best_val_auc
    
    def estimate_noise(self,tensor_val_mask):
        with torch.no_grad():
            val_y=self.tensor_y[tensor_val_mask]
            model=self.pretrain_model
            logits,_ = model(self.tensor_x,drop_out=0)
            logits=torch.softmax(logits,dim=1)
            test_mask_logits = logits[tensor_val_mask]
            values,ids=test_mask_logits[:,1].sort(descending=True)
            th=values[math.ceil(sum(val_y))]
            val_pos_mask=tensor_val_mask*(self.tensor_y==1)
            val_pos_pred=logits[val_pos_mask][:,1]>th
            return 1-(sum(val_pos_pred)/len(val_pos_pred)).item()
    def cores_train(self,tensor_train_mask,tensor_val_mask,tensor_test_mask,args,tolerance=100):
        pretrain_cri=torch.nn.CrossEntropyLoss(weight=self.pos_weight)
        pretrain_opt=optim.Adam(self.pretrain_model.parameters(),lr=args.pretrain_lr, weight_decay=1e-7)
        #tensor_train_mask=tensor_train_mask+tensor_val_mask
        epochs=1000
        loss_all = np.zeros((self.sample_size,epochs))
        loss_div_all = np.zeros((self.sample_size,epochs))
        loss_history = []
        val_auc_history = []
        train_auc_history=[]
        test_auc_history=[]
        self.pretrain_model.train()
        all_mask=tensor_train_mask+tensor_val_mask
        val_neg_mask=tensor_val_mask*(self.tensor_y==1)
        stop_increasing=0
        v_list = np.zeros(sum(all_mask))## sieve pass flag for all instances
        val_acc=1
        max_auc=0
        for epoch in tqdm(range(epochs)):
            self.pretrain_model.train()
            logits,weight = self.pretrain_model(self.tensor_x,drop_out=self.dropout)  
            self.att_weight=weight
            train_mask_logits = logits[tensor_train_mask]   
            loss = pretrain_cri(train_mask_logits, self.tensor_y[tensor_train_mask])
            if epoch==299:
                print('epoch 300')
            mask_logits=logits[tensor_train_mask]   
            noise_star=self.estimate_noise(tensor_val_mask)
            loss,loss_v= loss_cores(epoch,mask_logits,self.tensor_y[tensor_train_mask],tensor_train_mask,loss_all,loss_div_all,device=args.device,pos_weight=self.pos_weight,noise_prior=noise_star*0.5)   
            pretrain_opt.zero_grad()
            loss.backward()     
            pretrain_opt.step()    
            train_acc, _, _ = self.test(tensor_train_mask,pretrain=True,drop_out=self.dropout)    
            noise_star=self.estimate_noise(tensor_val_mask)
            loss_history.append(loss.item())
            if train_acc>=0.95:
                break
            val_acc, _, _ = self.test(tensor_val_mask,drop_out=0.0,pretrain=True)    
            test_acc, _, _ = self.test(tensor_test_mask,drop_out=0.0,pretrain=True)
           
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
    
    def pretrain(self,tensor_train_mask,tensor_val_mask,epochs,args):
        pretrain_cri=torch.nn.CrossEntropyLoss(weight=self.pos_weight)
        pretrain_opt=optim.Adam(self.pretrain_model.parameters(),lr=args.pretrain_lr, weight_decay=1e-7)
        #tensor_train_mask=tensor_train_mask+tensor_val_mask
        loss_all = np.zeros((self.sample_size,epochs))
        loss_div_all = np.zeros((self.sample_size,epochs))
        loss_history = []
        train_auc_history=[]
        self.pretrain_model.train()
        all_mask=tensor_train_mask+tensor_val_mask
        val_neg_mask=tensor_val_mask*(self.tensor_y==1)
        stop_increasing=0
        v_list = np.zeros(sum(all_mask))## sieve pass flag for all instances
        val_acc=1
        max_auc=0
        for epoch in tqdm(range(epochs)):
            self.pretrain_model.train()
            logits,weight = self.pretrain_model(self.tensor_x,drop_out=self.dropout)  
            self.att_weight=weight
            train_mask_logits = logits[tensor_train_mask]   
            loss = pretrain_cri(train_mask_logits, self.tensor_y[tensor_train_mask])
            if epoch==299:
                print('epoch 300')
            mask_logits=logits[all_mask]   
            noise_star=self.estimate_noise(tensor_val_mask)
            _,loss_v= loss_cores(epoch,mask_logits,self.tensor_y[all_mask],all_mask,loss_all,loss_div_all,device=args.device,pos_weight=self.pos_weight,noise_prior=noise_star*1.5)   
            pretrain_opt.zero_grad()
            loss.backward()     
            pretrain_opt.step()    
            train_acc, _, _ = self.test(tensor_train_mask,pretrain=True,drop_out=self.dropout)    
            noise_star=self.estimate_noise(tensor_val_mask)
            loss_history.append(loss.item())
            if train_acc>=0.95:
                break
            # if train_acc>max_auc:
            #     max_auc=train_acc
            #     v_list=loss_v
            # else:
            #     break
            train_auc_history.append(train_acc.item())
           
            # class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(2)]
            # noise_prior_delta = np.array(class_size_noisy)
            # noise_prior_cur = noise_prior*len(train_y) - noise_prior_delta
            # noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)
            if epoch%50==0 and self.printFlag:
                print("Epoch {:03d}: Loss {:.4f}, TrainAuc {:.4},estimated noise rate {:.4} , sieve-passing example:{}".format(
                    epoch, loss.item(), train_acc.item(),noise_star,sum(loss_v)))
        
        
        return loss_history,train_auc_history,loss_v
    
    def train(self,tensor_train_mask,tensor_val_mask,tensor_test_mask,args,Bayesian_T=None,tolerance=100):
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
        m,n=self.tensor_x[0].shape
        train_m=sum(tensor_train_mask.detach().cpu().numpy())
        if Bayesian_T is not None:
            Bayesian_T.eval()
            tm = Bayesian_T(self.sum_adj,self.tensor_x[0])
            #tm = Bayesian_T(self.tensor_x[0])
            if not args.fix_half:
                tm=tm[:,2:4]
            tm=F.softmax(tm[tensor_train_mask],dim=1)
            tm=torch.cat((torch.FloatTensor([[1,0]]*train_m).to(args.device),tm),dim=1).view(train_m,2,2)
            tm=tm.detach()
        for epoch in range(args.max_steps):
            self.model.train()
            logits,weight = self.model(self.tensor_x,drop_out=self.dropout)  
            self.att_weight=weight
            train_mask_logits = logits[tensor_train_mask]  
            ### loss correction
            if Bayesian_T is not None:
                # if revision == True:
                #     tm = tools.norm(tm + delta)
                noisy_post = torch.bmm(train_mask_logits.unsqueeze(dim=1),tm).squeeze()
                
                loss = criterion(noisy_post, train_y)
            else:
                loss=criterion(train_mask_logits,train_y)
            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()    
            train_acc, _, _ = self.test(tensor_train_mask,drop_out=self.dropout)     
            val_acc, _, _ = self.test(tensor_val_mask,drop_out=0.0)    
            test_acc, _, _ = self.test(tensor_test_mask,drop_out=0.0)
           
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
    def test(self,mask,metric='auc',pretrain=False,drop_out=0):
        
            
        if pretrain:
            model=self.pretrain_model
        else:
            model=self.model
        model.eval()
        with torch.no_grad():
            
            logits,_ = model(self.tensor_x,drop_out=drop_out)
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
