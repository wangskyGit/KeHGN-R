import torch
import torch.nn.functional as F
from model import tmGCN,NN
from torch import optim

def transition_matrix_learn(distilled_label:torch.LongTensor,noised_label:torch.LongTensor,X:torch.FloatTensor,distilled_mask,adj,args):
    """
    learning for transition matrix
    """
    device=args.device
    out_dim=2 if args.fix_half else 4
    TM_gcn=tmGCN(input_dim=X.size()[1],hidden_dim=500,out_dim=out_dim)
    #TM_gcn=NN(input_dim=X.size()[1],hidden_dim=500,out_dim=out_dim)
    criterion = torch.nn.NLLLoss(reduction='none')
    
    bayes_pos_mask=distilled_label==1
    optimizer = optim.Adam(TM_gcn.parameters(),lr=args.tm_lr, weight_decay=1e-6)
    distilled_onehot=F.one_hot(distilled_label.to(device)).float()# m*2
    distilled_X=X[distilled_mask]
    m=sum(distilled_mask)
    TM_gcn.train()
    TM_gcn.cuda(device=device)
    nit=0
    min_loss=1000
    for epoch in range(args.tm_steps):
        noisy_class_post = torch.zeros((sum(distilled_mask), 2))
        tm=TM_gcn(adj,X)# m*2
        #tm=TM_gcn(X)
        if not args.fix_half:
            tm=tm[distilled_mask].view(m,2,2)
            tm=F.softmax(tm,dim=2)
        else:
            tm=F.softmax(tm[distilled_mask],dim=1)
            tm=torch.cat((torch.FloatTensor([[1,0]]*m).to(device),tm),dim=1).view(m,2,2)
        noisy_class_post=torch.bmm(distilled_onehot.unsqueeze(dim=1),tm).squeeze()
         # fixed the value for class 1 in transition matrix
        # for j in range(sum(distilled_mask)):
        #     bayes_label_one_hot = distilled_onehot[j].unsqueeze(dim=0) # 1*2 bayes optimal label
        #     tm_j=torch.cat((torch.FloatTensor([1,0]).to(device),tm[j])).view(2,2) # 2*2 transition matrix
        #     noisy_class_post_temp = bayes_label_one_hot.mm(tm_j) # 1*2 noisy label
        #     noisy_class_post[j, :] = noisy_class_post_temp
        noisy_class_post = torch.log(noisy_class_post+1e-12)
        loss = criterion(noisy_class_post.to(device), noised_label.to(device))
        loss = torch.mean(loss)# * (distilled_label.to(device)*3+1)) # use bayes optimal label as weighting 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if nit>10:
            break
        if min_loss<loss.item():
            nit+=1
            
            print('loss increase, break; Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
        else:
            nit=0
            torch.save(TM_gcn.state_dict(), args.save_path + '/' + 'BayesianT.pth')
            min_loss=loss.item()
        if epoch % 5 ==0:
            print('Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
    TM_gcn.load_state_dict(torch.load(args.save_path + '/' + 'BayesianT.pth'))
    tm=TM_gcn(adj,X)
    #tm=TM_gcn(X)
    if not args.fix_half:
            tm=tm[distilled_mask].view(m,2,2)
            tm=F.softmax(tm,dim=2)
    else:
        tm=F.softmax(tm[distilled_mask],dim=1)
        tm=torch.cat((torch.FloatTensor([[1,0]]*m).to(device),tm),dim=1).view(m,2,2)
    print('Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
    torch.save(TM_gcn.state_dict(), args.save_path + '/' + 'BayesianT.pth')

    return tm,TM_gcn
def bltm(distilled_label:torch.LongTensor,noised_label:torch.LongTensor,X:torch.FloatTensor,distilled_mask,args):
    """
    learning for transition matrix
    """
    args.fix_half=False
    device=args.device
    out_dim=4 
    TM_nn=NN(input_dim=X.size()[1],hidden_dim=500,out_dim=out_dim)
    criterion = torch.nn.NLLLoss(reduction='none')
    bayes_pos_mask=distilled_label==1
    optimizer = optim.Adam(TM_nn.parameters(),lr=args.tm_lr, weight_decay=1e-6)
    distilled_onehot=F.one_hot(distilled_label.to(device)).float()# m*2
    distilled_X=X[distilled_mask]
    m=sum(distilled_mask)
    TM_nn.train()
    TM_nn.cuda(device=device)
    nit=0
    min_loss=1000
    for epoch in range(args.tm_steps):
        noisy_class_post = torch.zeros((sum(distilled_mask), 2))
        tm=TM_nn(X)# m*2
        #tm=TM_gcn(X)
        tm=tm[distilled_mask].view(m,2,2)
        tm=F.softmax(tm,dim=2)
        noisy_class_post=torch.bmm(distilled_onehot.unsqueeze(dim=1),tm).squeeze()
        noisy_class_post = torch.log(noisy_class_post+1e-12)
        loss = criterion(noisy_class_post.to(device), noised_label.to(device))
        loss = torch.mean(loss)# * (distilled_label.to(device)*3+1)) # use bayes optimal label as weighting 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if nit>10:
            break
        if min_loss<loss.item():
            nit+=1
            print('loss increase, break; Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
        else:
            nit=0
            torch.save(TM_nn.state_dict(), args.save_path + '/' + 'BayesianT.pth')
            min_loss=loss.item()
        if epoch % 5 ==0:
            print('Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
    TM_nn.load_state_dict(torch.load(args.save_path + '/' + 'BayesianT.pth'))
    tm=TM_nn(X)
    tm=tm[distilled_mask].view(m,2,2)
    tm=F.softmax(tm,dim=2)
    print('Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
    torch.save(TM_nn.state_dict(), args.save_path + '/' + 'BayesianT.pth')

    return tm,TM_nn