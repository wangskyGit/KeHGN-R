from layer import mwGCN_wot,gcn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from util import attention_sum,attention_sum2
from layer import mwGCN

class KeGCN(nn.Module):
    """
    The full model
    model with two layer mwGCN, reltion attention and embedding attention
    """
    def __init__(self, adj_list,device,input_dim=343,embed_dim=1024,att_activate=None,use_kg=False,kg_dim=None,share=False):
        super(KeGCN, self).__init__()
        self.rel_num=len(adj_list)
        self.embed_dim=embed_dim
        self.gcn1 = mwGCN(input_dim, embed_dim,device=device,rel_num=self.rel_num,att_activate=att_activate,use_att=False,share=share)
        self.gcn2 = mwGCN(embed_dim, embed_dim,device=device,rel_num=self.rel_num,use_att=True,att_activate=att_activate,share=share)
        self.use_kge=use_kg
        self.att_activate=att_activate
        if use_kg:
            #self.kg_mlp=nn.Linear(kg_dim,embed_dim)
            #self.rel_num-=1
            self.kg_gcn1 = mwGCN(kg_dim, embed_dim,device=device,rel_num=self.rel_num,use_att=False,att_activate=att_activate,share=share)
            self.kg_gcn2 = mwGCN(embed_dim, embed_dim,device=device,rel_num=self.rel_num,use_att=True,att_activate=att_activate,share=share)
            
            # self.att_liner=nn.Linear(embed_dim,embed_dim)
            # self.att_q=nn.Parameter(torch.Tensor(embed_dim,1))
            # init.kaiming_uniform_(self.att_q)
            self.kg_att_weight=nn.Parameter(torch.Tensor(embed_dim,1))
            self.kg_att_bias=nn.Parameter(torch.Tensor(1))
            init.kaiming_uniform_(self.kg_att_weight)
            init.zeros_(self.kg_att_bias)
                
        #self.mlp=nn.Linear(embed_dim,embed_dim)
        self.output=nn.Linear(embed_dim,2)
        self.adj_list=adj_list
        self.layer_norm=nn.LayerNorm(normalized_shape= embed_dim)
    def forward(self,features,drop_out=0.5):
        self.dropout = nn.Dropout(p=drop_out)
        feature=features[0]
        h1=self.gcn1(self.adj_list,[feature for _ in range(self.rel_num)])
        h1=[F.relu(self.dropout(r)) for r  in h1]
        #h1=[feature for _ in range(self.rel_num)]
        h2,w=self.gcn2(self.adj_list,h1)
        h2 =F.relu(self.dropout(h2))
        att_value={'rel_weight':w.cpu().detach().numpy()}
        if self.use_kge:
            kge=features[-1]
            kge_h1=self.kg_gcn1(self.adj_list,[kge for _ in range(self.rel_num)])
            kge_h1=[F.relu(self.dropout(r)) for r in kge_h1]
            #kge_h1=[kge for _ in range(self.rel_num)]
            kge_h2,kge_w=self.kg_gcn2(self.adj_list,kge_h1)
            kge_h2 = F.relu(self.dropout(kge_h2))
            kge_h2=self.layer_norm(kge_h2)
            h2=self.layer_norm(h2)
            h2,emb_w=attention_sum([h2,kge_h2],self.kg_att_weight,self.kg_att_bias,self.att_activate,device=h2.device)
            #h2,emb_w=attention_sum2([h2,kge_h2],self.att_q,self.att_liner,self.att_activate,device=h2.device)
            att_value['emb_weight']=emb_w.cpu().detach().numpy()
            att_value['kge_weight']=kge_w.cpu().detach().numpy()
        #h2=torch.sigmoid(self.mlp(h2))    
        logits=self.output(h2)
        return logits,att_value


class GCN(nn.Module):
    def __init__(self,hidden_dim=1000,input_dim=343,out_dim=2*2,drop_out=0.5) -> None:
        super(GCN,self).__init__()
        self.gcn1=gcn(input_dim=input_dim,output_dim=hidden_dim)
        #self.gcn2=gcn(input_dim=hidden_dim,output_dim=hidden_dim)
        self.fcl=nn.Linear(in_features=hidden_dim,out_features=out_dim)
        self.dropout=nn.Dropout(p=drop_out)
        self.layer_norm=nn.LayerNorm(normalized_shape= hidden_dim)
    def forward(self,adj,features) -> torch.tensor:
        h1= F.relu(self.gcn1(adj,features))
        if self.training:
            h1=self.dropout(h1)
        # h2=F.relu(self.gcn2(adj,h1))
        # if self.training:
        #     h2=self.dropout(h2)
        h2=self.layer_norm(h1)
        return self.fcl(h2)
class tmGCN(nn.Module):
    def __init__(self,hidden_dim=1000,input_dim=343,out_dim=2*2,drop_out=0.4) -> None:
        super(tmGCN,self).__init__()
        
        self.gcn1=gcn(input_dim=input_dim,output_dim=hidden_dim)
        self.gcn2=gcn(input_dim=hidden_dim,output_dim=out_dim)
        self.mlp=nn.Linear(in_features=hidden_dim,out_features=out_dim)
        self.dropout=nn.Dropout(p=drop_out)
        self.layer_norm=nn.LayerNorm(normalized_shape= hidden_dim)
    def forward(self,adj,features) -> torch.tensor:
        h1=F.relu(self.gcn1(adj,features))
        if self.training:
            h1=self.dropout(h1)
        h1=self.layer_norm(h1)
        h2=self.mlp(h1)
        if self.training:
            h2=self.dropout(h2)
        return h2


class NN(nn.Module):
    def __init__(self,hidden_dim=1000,input_dim=343,out_dim=2,drop_out=0.5) -> None:
        super(NN,self).__init__()
        self.nn1=nn.Linear(in_features=input_dim,out_features=hidden_dim)
        self.nn2=nn.Linear(in_features=hidden_dim,out_features=out_dim)
        self.dropout=nn.Dropout(p=drop_out)
        self.layer_norm=nn.LayerNorm(normalized_shape= hidden_dim)
    def forward(self,X):
        h1=torch.sigmoid(self.nn1(X))
        if self.training:
            h1=self.dropout(h1)
        h2=self.nn2(h1)
        if self.training:
            h2=self.dropout(h2)
        return h2
class KeGN(nn.Module):
    """
    model with two layer mwGCN and without attnetion mechnism
    """
    def __init__(self, adj_list,device,input_dim=343,embed_dim=1024,att_activate=None,use_kg=False,kg_dim=None,share=False):
        super(KeGN, self).__init__()
        self.rel_num=len(adj_list)
        self.embed_dim=embed_dim
        self.device=device
        self.gcn1 = mwGCN_wot(input_dim, embed_dim,device=self.device,rel_num=self.rel_num,use_att=True,share=share)
        #self.gcn2 = mwGCN_wot(embed_dim , embed_dim,rel_num=self.rel_num,use_att=True,share=share)
        self.use_kge=use_kg
        self.att_activate=att_activate
        if use_kg:
            self.kg_gcn1 = mwGCN_wot(kg_dim, embed_dim,device=self.device,rel_num=self.rel_num,use_att=True,share=share)
            #self.kg_gcn2 = mwGCN_wot(embed_dim , embed_dim,rel_num=self.rel_num,use_att=True,share=share)
        self.nn1=nn.Linear(in_features=embed_dim,out_features=embed_dim)
        self.output=nn.Linear(embed_dim,2)
        self.adj_list=adj_list
        self.layer_norm=nn.LayerNorm(normalized_shape= embed_dim)
    def forward(self,features,drop_out=0.5):
        self.dropout = nn.Dropout(p=drop_out)
        feature=features[0]
        h1=self.gcn1(self.adj_list,[feature for _ in range(self.rel_num)])
        # h1=[F.relu(self.dropout(r)) for r  in h1]
        # h2=self.gcn2(self.adj_list,h1)
        h2 = F.relu(self.dropout(h1))
        if self.use_kge:
            kge=features[-1]
            kge_h1=self.kg_gcn1(self.adj_list,[kge for _ in range(self.rel_num)])
            #kge_h1=[F.relu(self.dropout(r)) for r in kge_h1]
            # kge_h2=self.kg_gcn2(self.adj_list,kge_h1)
            kge_h2 = F.relu(self.dropout(kge_h1))
            h2=self.layer_norm(h2)
            kge_h2=self.layer_norm(kge_h2)
            h2=h2+kge_h2
        h2=torch.sigmoid(self.nn1(h2))
        logits=self.output(h2)
        return logits,None

