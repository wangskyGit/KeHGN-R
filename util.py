from distutils.log import error
from json import load
import scipy.sparse as sp
import pandas as pd
import numpy as np  
import torch
import csv
import random
from sklearn.preprocessing import minmax_scale
from math import ceil

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def normalize_adj(adjacency):
    """row normalization"""
    adjacency += 0.0001*sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return (d_hat.dot(adjacency).dot(d_hat)+sp.eye(adjacency.shape[0])).tocoo()

def attention_sum(tensors,att_weight,att_bias,att_activate,device):
    graph_emb=[torch.mean(fea,axis=0) for fea in tensors]
    weight=att_activate(torch.tensor([torch.mm(emb.unsqueeze(0),att_weight.to(device))+att_bias.to(device) for emb in graph_emb],device=device))
    weight=torch.softmax(weight,dim=0)
    rel_num=len(tensors)
    for i in range(rel_num):
        if i==0:
            real_output=weight[0]*tensors[0]
        else:
            real_output+=weight[i]*tensors[i]
    return real_output,weight
def attention_sum2(tensors,att_q,att_linear,att_activate,device):
    w=[torch.mm(att_activate(att_linear(h)),att_q) for h in tensors]
    w=torch.tensor([h.squeeze().mean(dim=0) for h in w],device=device)
    w=torch.softmax(w,dim=0)
    rel_num=len(tensors)
    for i in range(rel_num):
        if i==0:
            real_output=w[0]*tensors[0]
        else:
            real_output+=w[i]*tensors[i]
    return real_output,w
def hide_attr(X:np.ndarray,per=0.2,seed=5):
    f_size=X.shape[1]
    random.seed(seed)
    col=random.sample(range(f_size), int(f_size*(per)))
    X[:,col]=np.random.random((len(X),len(col)))
    return X   
def normalize(df:pd.DataFrame,result_df:pd.DataFrame=None):
    data=df.loc[:,df.columns.drop(['year'])]
    data=minmax_scale(data)
    df.loc[:,df.columns.drop(['year'])]=data
    df.reset_index(inplace=True)
    # result_df=pd.concat([result_df,df],axis=0)
    return df
def load_data(args,seed=5):
    dataset=args.dataset
    hid=args.hide
    mw_rpt=sp.load_npz('./dataset/'+dataset+'/rpt.npz')
    mw_sc=sp.load_npz('./dataset/'+dataset+'/sc.npz')
    mw_sdse=sp.load_npz('./dataset/'+dataset+'/sdse.npz')
    attribute=pd.read_csv('./dataset/'+dataset+'/fv.csv')
    good_comp=pd.read_csv('./dataset/good_comp_{}.csv'.format(args.dataset))
    # 
    X=attribute.drop(['IndustryCode','EquityNatureID','Stknmec','label','Stkcd','year'],axis=1)
    ### fill missing with average
    for column in X.columns:
        if X[column].isna().sum() > 0:
            mean_val = X[column].mean()
            X[column].fillna(mean_val, inplace=True)
    # X=X.groupby(['year']).apply(normalize)
    # X=X.reset_index(drop=True)
    # index=X['index']
    #X=X.drop(['index','year'],axis=1)
    X=X.values
    # index_df=attribute.loc[index][['Stkcd','year','label']]
    index_df=attribute[['Stkcd','year','label']]
    index_df['index']=list(range(0,len(attribute)))
    tmp=index_df.loc[index_df['label']==1]
    # high_conf_comp=index_df.merge(good_comp,how='right',on=['Stkcd','year'])[['Stkcd','year','label','index']]
    # high_conf_comp=pd.concat([tmp,high_conf_comp],axis=0).sort_values(by='index')
    high_conf_comp=index_df.loc[index_df.apply(lambda x:x['year']<=2010 or x['label']==1,axis=1)]
    if args.dataset=='entre':
        high_conf_comp=index_df.loc[index_df.apply(lambda x:x['year']<=2012 or x['label']==1,axis=1)]
    y=index_df['label'].values
    X=minmax_scale(X)
    if hid !=0:
        X=hide_attr(X,per=hid,seed=seed)
    #y=attribute['label'].values
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y)
    if 'Ke' in args.model or args.model=='COREAS':
        ### get knowledge embedding
        node_embedding=np.load('./FKG/{}/TransE_l2/TransE_l2_entity.npy'.format(args.dataset))
        node_dict=pd.read_csv('./FKG/{}/entities.tsv'.format(args.dataset),header=None,sep='\t',quoting=csv.QUOTE_NONE,index_col=1)
        node_embedding=pd.DataFrame(node_embedding)
        node_embedding['node']=node_dict.index
        fv=pd.DataFrame(index_df.apply(lambda x : str(int(x['year']))+'.comp.'+str(int(x['Stkcd'])),axis=1),columns=['node'])
        kge=fv.merge(node_embedding,on='node',how='left').drop(['node'],axis=1)
        #kge=pd.DataFrame(np.load('./FKG/TransE_FKG/comp_kge.npy'))
        
        for column in kge.columns:
            if kge[column].isna().sum() > 0:
                mean_val = kge[column].mean()
                kge[column].fillna(mean_val, inplace=True)
        kge=torch.FloatTensor(kge.values)
    else:
        kge=None
    return features,labels,index_df,kge,[mw_rpt,mw_sc,mw_sdse],high_conf_comp

def corrupt_pos_label(labels,train_ids,test_ids,neg_2_pos=6,noise_rate=0.2,seed=5,yearly=False,index_df:pd.DataFrame=None,gap_th=3):
    """
    random corrupt positive labels in the training dataset into negative
    """
    if noise_rate==0:
        return labels,train_ids
    random.seed(seed)
    pos_ids= np.array(range(len(labels)))[labels==1]
    neg_ids= np.array(range(len(labels)))[labels==0]
    if yearly and index_df is not None:
        val_year=pd.read_csv('./dataset/violation_year.csv')
        val_year=val_year.loc[val_year[['Stkcd','year']].drop_duplicates().index]
        index_df=index_df.merge(val_year,how='left',on=['Stkcd','year'])
        index_df_pos=index_df.loc[index_df['label']==1]
        candicate_index=index_df_pos.loc[index_df_pos['gap_year']>=gap_th].index.values
        corrupt_index=list(set(candicate_index)-set(test_ids))
        print('corrput frauds with gap year >={}, total number:{}'.format(gap_th,len(corrupt_index)))
    else:
        candicate_index=list(set(pos_ids)-set(test_ids))
        corrupt_index=random.sample(candicate_index,ceil(len(candicate_index)*noise_rate))
    # hidden_condi=list(set(neg_ids)&set(train_ids))
    # hidden_train_ids=random.sample(hidden_condi,ceil(len(corrupt_index)*neg_2_pos))
    # train_ids=list(set(train_ids)-set(hidden_train_ids))
    cor_label=labels.clone()
    cor_label[corrupt_index]=0
    return cor_label,np.array(train_ids)

def load_data_pyG(dataset='full'):
    features,labels,kge,[mw_rpt,mw_sc,mw_sdse]=load_data(dataset)
    data = HeteroData()
    data['comp'].x=features
    data['comp','rpt','comp'].edge_index,data['comp','rpt','comp'].edge_attr=np.stack((mw_rpt.tocoo().row,mw_rpt.tocoo().col)),mw_rpt.tocoo().data
    data['comp','sdse','comp'].edge_index,data['comp','sdse','comp'].edge_attr=np.stack((mw_sdse.tocoo().row,mw_sdse.tocoo().col)),mw_sdse.tocoo().data
    data['comp','sc','comp'].edge_index,data['comp','sc','comp'].edge_attr=np.stack((mw_sc.tocoo().row,mw_sc.tocoo().col)),mw_sc.tocoo().data
    

if  __name__=='__main__':
    load_data_pyG()