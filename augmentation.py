import torch
import numpy as np
from copy import deepcopy
from torch_geometric.data import Batch
import torchvision.transforms as imgtransforms
from dataset import GaussianBlur

def drop_nodes(data):

    node_num = data.num_nodes
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index

    return data


def permute_edges(data):

    node_num = data.num_nodes
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()
    idx_add = np.random.choice(node_num, (permute_num, 2))
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def subgraph(data):

    node_num = data.num_nodes
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)
    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index

    return data


def mask_nodes(data):

    if data.x is None:
        return data
    
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data


def aug_graph(batch_data, n):
    batch_data_aug = []

    for data in batch_data.to_data_list():

        if n == 0:
            data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
            data_aug = subgraph(deepcopy(data))
        elif n == 2:
            data_aug = mask_nodes(deepcopy(data))
        elif n == 3:
            data_aug = permute_edges(deepcopy(data))
        
        node_num = data.num_nodes
        edge_idx = data_aug.edge_index.numpy()
        _, edge_num = edge_idx.shape
        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
        node_num_aug = len(idx_not_missing)
        if node_num_aug == 0:
            batch_data_aug.append(data)
            continue
        if data.x is None:
            data_aug.x =None
        else:
            data_aug.x = data_aug.x[idx_not_missing]
        data_aug.num_nodes = node_num_aug
        idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}    
        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]    
        data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
        batch_data_aug.append(data_aug)
 
    return batch_data, Batch().from_data_list(batch_data_aug)


def aug_img(batch_data, n):      
    return batch_data[0][0], batch_data[0][1]
