import numpy as np  
from sklearn.model_selection import train_test_split
from util import load_data,normalize_adj,sample_mask,corrupt_pos_label
#import matplotlib.pyplot as plt
import random
import pandas as pd
from exp import ExpModel
from baseline import NNBaseline,GCNBaseline
import argparse
import math

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing KeHCN',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true',default=True, help='use GPU')
    parser.add_argument('--share',action='store_true',default=False,help='whether to share parameters in different subgraphs during gcn')
    parser.add_argument('--att_act',type=str,default='Tanh',help='activation of attention calculation, LeakyRelu or Tanh')
    parser.add_argument('--gcn', action='store_true')
    parser.add_argument('--dataset', type=str, default='main')
    parser.add_argument('--model', default='KeHGNN', type=str)
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-plr', '--pretrain_lr', default=0.0001, type=float)
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--pretrain_steps',default=300,type=int)
    parser.add_argument('--tolerance', default=300, type=int)
    parser.add_argument('-w','--pos_weight', default=6.0, type=float,help='weight for positive class')
    parser.add_argument('-dp','--drop_out', default=0.0, type=float,help='drop out rate')
    parser.add_argument('--nke', action='store_true', help='diable knowledge embedding in model')
    parser.add_argument('--hide',type=float,default=0.0)
    parser.add_argument('--tm_steps',type=int,default=50)
    parser.add_argument('--tm_lr',type=float,default=0.0001)
    parser.add_argument('--save_path',type=str,default='./model')
    parser.add_argument('--fix_half',action='store_true',default=True,help='fix half of the transition matrix ')
    parser.add_argument('--noise_rate',default=0.1,type=float,help='randomly corruptted noise rate')
    parser.add_argument('--device',default=0,type=int)
    return parser.parse_args(args)


if __name__=='__main__':
    args=parse_args()
    args.fix_half=True
    re=pd.DataFrame(columns=['posw','dr','val_auc','testauc'])
    run=0
    features,labels,index_df,kge,sub_graphs,high_conf_comp=load_data(args)
    m,n=features.shape
    mw_list = [normalize_adj(adj) for adj in sub_graphs]
    #adj_list = [normalize_adj(sp.coo_matrix(np.where(adj.todense()!=0,1,0), dtype=np.float32)) for adj in [rpt,sc,sdse]]
    if  'Ke' in args.model or args.model=='COREAS':
        inits=[features,kge]
        #inits=[kge,features]
    else:
        inits= features

    for nr in [0]:
        args.noise_rate=nr
        print('noise rate = {}'.format(nr))
        for seed in [9,99,999,9999,99999]:# 6 66 666 6666 66666 9 99 999 9999 99999
            posw_re_val=[]
            posw_re_test=[]
            random.seed(seed)
            # if nr==0:
            #     args.model='KeHGN'
            ### train val test split (6:2:2) 
            pos_ratio=sum(labels).item()/(len(labels))
            pos_test=random.sample(list(high_conf_comp['index'].values[high_conf_comp['label']==1]),math.ceil(m*0.2*pos_ratio))
            neg_test=random.sample(list(high_conf_comp['index'].values[high_conf_comp['label']==0]),math.ceil(m*0.2*(1-pos_ratio)))
            test_ids=np.array(pos_test+neg_test)
            train_ids=np.array(list(set(range(m))-set(test_ids)))
            #train_ids,test_ids=train_test_split(list(range(m)),test_size=0.2,stratify=labels.numpy())
            corrupted_labels,train_ids=corrupt_pos_label(labels,train_ids,test_ids,noise_rate=nr,neg_2_pos=(1-pos_ratio)/pos_ratio,seed=seed,
                                                         yearly=False)
            #train_mask_row=sample_mask(train_ids,m)
            train_ids,val_ids=train_test_split(train_ids,test_size=0.25,stratify=labels[train_ids],random_state=seed)
            #test_ids,val_ids=train_test_split(test_ids,test_size=0.5,stratify=labels[test_ids],random_state=seed)
            test_mask = sample_mask(test_ids, m)
            train_mask = sample_mask(train_ids, m)
            val_mask = sample_mask(val_ids, m)
            print('test size:{}, test positive ratio:{}, train size{}, train positive ratio:{}, val size {} val postive ratio {}'.format(
                sum(test_mask),sum(labels[test_mask])/sum(test_mask),sum(train_mask),sum(corrupted_labels[train_mask])/sum(train_mask),sum(val_mask),sum(corrupted_labels[val_mask])/sum(val_mask)
            ))
            for posw in [4.0,6.0,8.0]:#4,6,8
                for dr in [0,0.5]:#0,0.5
                    print('Try positive weight {}, drop out function:{}'.format(posw,dr))
                    args.pos_weight=posw
                    args.drop_out=dr
                    if args.model=='GCN':
                        exp=GCNBaseline(print_flag=True,early_stopping=True,args=args,seed=seed)
                        loss_history,train_auc_history,val_auc_history,test_auc_history,best_e,best_val_auc=exp.fit(mw_list,inits,corrupted_labels,train_mask,val_mask,test_mask,args=args)
                    elif args.model=='NN':
                        exp=NNBaseline(print_flag=True,early_stopping=True,args=args,seed=seed)
                        loss_history,train_auc_history,val_auc_history,test_auc_history,best_e,best_val_auc=exp.fit(inits,corrupted_labels,train_mask,val_mask,test_mask,args=args)
                    else:
                        exp=ExpModel(print_flag=True,early_stopping=True,args=args,seed=seed)
                        loss_history,train_auc_history,val_auc_history,test_auc_history,best_e,best_val_auc=exp.fit(mw_list,inits,corrupted_labels,train_mask,val_mask,test_mask,args=args)
                # exp=GCNBaseline(print_flag=True,early_stopping=True,args=args,seed=seed)
                    print('Best result on validation set: epoch:{},best val auc:{},test auc:{}'.format(best_e,best_val_auc,test_auc_history[best_e]))
                    posw_re_val.append(best_val_auc)
                    posw_re_test.append(test_auc_history[best_e])
                    re.loc[run,:]=[posw,dr,best_val_auc,test_auc_history[best_e]]
                    re.to_csv('{}_{}_random_1.5ns.csv'.format(args.model,args.dataset))
                    run+=1
