import torch
import torch.nn as nn
import torch.nn.init as init
from util import attention_sum,attention_sum2
import math
class mwGCN(nn.Module):
    def __init__(self, input_dim, output_dim,device,rel_num=1, use_bias=True,use_att=True,att_activate=None,share=False):
        super(mwGCN, self).__init__()
        self.device = device 
        self.input_dim = input_dim
        self.use_att=use_att
        self.rel_num=rel_num
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.share=share
        if self.use_att:
            self.att_weight=nn.Parameter(torch.Tensor(output_dim,1))
            self.att_bias=nn.Parameter(torch.Tensor(1))
            # self.att_linear=nn.Linear(output_dim,output_dim)
            # self.att_q=nn.Parameter(torch.Tensor(output_dim,1))
            self.att_activate=att_activate
        if share:
            self.weight=nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            for i in range(rel_num):
                exec('self.weight{}=nn.Parameter(torch.Tensor(input_dim, output_dim))'.format(i))
                if self.use_bias:
                    exec('self.bias{}=nn.Parameter(torch.Tensor(output_dim))'.format(i))
                else:
                    self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_att:
            init.kaiming_uniform_(self.att_weight)
            init.zeros_(self.att_bias)
            #init.kaiming_uniform_(self.att_q, a=math.sqrt(5))
        if self.share:
            init.kaiming_uniform_(self.weight)
            init.zeros_(self.bias)
        else:
            for i in range(self.rel_num):
                exec('init.kaiming_uniform_(self.weight{})'.format(i))
                if self.use_bias:
                    exec('init.zeros_(self.bias{})'.format(i)
                    )

    def forward(self, adj_list, input_features):
        device = self.device
        output_list=[]
        for i in range(len(adj_list)):
            if not self.share:
                params=self.__dict__['_parameters']
                support=torch.mm(input_features[i], params['weight'+str(i)].to(device))
                output = torch.sparse.mm(adj_list[i], support)
                if self.use_bias:
                    output += params['bias'+str(i)].to(device)
                output_list.append(output)
            else:
                support=torch.mm(input_features[i], self.weight.to(device))
                #print(support)
                output = torch.sparse.mm(adj_list[i], support)
                if self.use_bias:
                    output += self.bias.to(device)
                output_list.append(output)
        if self.use_att:
            real_output,weight=attention_sum(tensors=output_list,att_weight=self.att_weight,att_bias=self.att_bias,att_activate=self.att_activate,device=device)
            #real_output,weight=attention_sum2(tensors=output_list,att_q=self.att_q,att_linear=self.att_linear,att_activate=self.att_activate,device=device)
            return real_output,weight
        else:
            return output_list

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class gcn(nn.Module):
    def __init__(self, input_dim, output_dim,use_bias=True):
        super(gcn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.weight=nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias=nn.Parameter(torch.Tensor(output_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        init.zeros_(self.bias)
                    
    def forward(self, adj, input):
        try:
            assert input.shape[1]==self.input_dim
        except:
            print('input dimension error, which is supposed to be {}'.format(self.input_dim))
        support=torch.mm(input, self.weight)
        #print(support)
        output = torch.sparse.mm(adj, support)
        if self.use_bias:
            output += self.bias
        return output
###without relation attention:
class mwGCN_wot(nn.Module):
    def __init__(self, input_dim, output_dim,device,rel_num=1, use_bias=True,use_att=True,share=False):
        super(mwGCN_wot, self).__init__()
        self.device = device 
        self.input_dim = input_dim
        self.use_att=use_att
        self.rel_num=rel_num
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.share=share
        if share:
            self.weight=nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            for i in range(rel_num):
                exec('self.weight{}=nn.Parameter(torch.Tensor(input_dim, output_dim))'.format(i))
                if self.use_bias:
                    exec('self.bias{}=nn.Parameter(torch.Tensor(output_dim))'.format(i))
                else:
                    self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.share:
            init.kaiming_uniform_(self.weight)
            init.zeros_(self.bias)
        else:
            for i in range(self.rel_num):
                exec('init.kaiming_uniform_(self.weight{})'.format(i))
                if self.use_bias:
                    exec('init.zeros_(self.bias{})'.format(i)
                    )
                    
    def forward(self, adj_list, input_features):
        device = self.device 
        output_list=[]
        for i in range(len(adj_list)):
            if not self.share:
                params=self.__dict__['_parameters']
                support=torch.mm(input_features[i], params['weight'+str(i)].to(device))
                output = torch.sparse.mm(adj_list[i], support)
                if self.use_bias:
                    output += params['bias'+str(i)].to(device)
                output_list.append(output)
            else:
                support=torch.mm(input_features[i], self.weight.to(device))
                #print(support)
                output = torch.sparse.mm(adj_list[i], support)
                if self.use_bias:
                    output += self.bias.to(device)
                output_list.append(output)
        if self.use_att:
            # directly sum up all subgraph output embedding 
            for j in range(len(output_list)):
                if j==0:
                    real_output=output_list[0]
                else:
                    real_output+=output_list[j]
            return real_output
        else:
            return output_list

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'