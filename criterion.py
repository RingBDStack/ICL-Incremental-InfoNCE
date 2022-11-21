import torch
from torch import nn


def loss_cal(x, x_aug, debug=False):
    T = 0.1
    batch_size, _ = x.size()
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x.norm(dim=1), x_aug.norm(dim=1))
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1))
    loss = - torch.log(loss).mean()
    if debug:
        print(x)
        print(x_aug)
        print(sim_matrix)
        print(pos_sim)
        print(loss)
        print()
    return loss


def loss_cal_ICL_old(old_x, old_x_aug, new_x, alpha):
    T = 0.1
    batch_size, _ = old_x.size()
    
    old2new_sim_matrix = torch.einsum('ik,jk->ij', old_x, new_x) / torch.einsum('i,j->ij', old_x.norm(dim=1), new_x.norm(dim=1))
    old2new_sim_matrix = torch.exp(old2new_sim_matrix / T)
    
    old2old_sim_matrix = torch.einsum('ik,jk->ij', old_x, old_x_aug) / torch.einsum('i,j->ij', old_x.norm(dim=1), old_x_aug.norm(dim=1))
    old2old_sim_matrix = torch.exp(old2old_sim_matrix / T)
    
    pos_sim = old2old_sim_matrix[range(batch_size), range(batch_size)]
       
    loss = (1-alpha) + alpha * (pos_sim + old2new_sim_matrix.sum(dim=1)) / (old2old_sim_matrix.sum(dim=1))
    loss = torch.log(loss).mean()

    return loss

def loss_cal_ICL_new(new_x, new_x_aug, old_x, new_num):
    T = 0.1
    batch_size, _ = new_x.size()

    new2new_sim_matrix = torch.einsum('ik,jk->ij', new_x, new_x_aug) / torch.einsum('i,j->ij', new_x.norm(dim=1), new_x_aug.norm(dim=1))
    new2new_sim_matrix = torch.exp(new2new_sim_matrix / T)
    
    pos_sim = new2new_sim_matrix[range(batch_size), range(batch_size)]
    
    if old_x is None:
        new2old_sim_matrix = torch.zeros([batch_size, 1]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        new2old_sim_matrix = torch.einsum('ik,jk->ij', new_x, old_x) / torch.einsum('i,j->ij', new_x.norm(dim=1), old_x.norm(dim=1))
        new2old_sim_matrix = torch.exp(new2old_sim_matrix / T)
        
    loss = pos_sim / (torch.cat((new2new_sim_matrix[:new_num, new_num], pos_sim[new_num:])) + new2new_sim_matrix[:, :new_num].sum(dim=1) + new2old_sim_matrix.sum(dim=1))
    loss = - torch.log(loss).mean()

    return loss
