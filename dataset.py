import torch
from torch_geometric.datasets import TUDataset, Reddit
import random
from torch_geometric.transforms import LocalDegreeProfile, Compose, NormalizeFeatures
import torchvision.datasets as imgdatasets
import torchvision.transforms as imgtransforms
from PIL import ImageFilter
from torch.utils.data import random_split


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def close_grad(data):
    data.x = data.x.detach()
    return data

def cut_feature(data):
    data.x = data.x.detach()[:,-5:]
    return data

class convert_label(object):
    
    def __init__(self, num):
        self.num = num
    
    def __call__(self, data):
        data.y += self.num
        return data


def process_data(mode, dataset, alpha):
    data = []

    path = 'data/' + mode + '/'
    
    if mode == 'cv':
        if dataset == 'ImageNet-2':            
            augmentation = [imgtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                            imgtransforms.RandomApply([imgtransforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            imgtransforms.RandomGrayscale(p=0.2),
                            imgtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                            imgtransforms.RandomHorizontalFlip(),
                            imgtransforms.ToTensor(),
                            imgtransforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            data = imgdatasets.ImageFolder(path+"ImageNet-2/", TwoCropsTransform(imgtransforms.Compose(augmentation)))
            n_class = 2
        elif dataset == 'MNIST-2':            
            augmentation = [imgtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                            imgtransforms.RandomApply([imgtransforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            imgtransforms.RandomGrayscale(p=0.2),
                            imgtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                            imgtransforms.RandomHorizontalFlip(),
                            imgtransforms.ToTensor(),
                            imgtransforms.Lambda(lambda x: x.repeat(3,1,1)),
                            imgtransforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            ori_data=imgdatasets.MNIST(path+"MNIST-2/", train=True, transform = TwoCropsTransform(imgtransforms.Compose(augmentation)), download=False)
            n_class = 2
    
        n_data = len(data)
        n_feature = -1
                
        split = round((1-alpha) * n_data)
        old_data, new_data = random_split(dataset=data, lengths=[split, n_data-split])
        print('old data {}, new data {}'.format(len(old_data), len(new_data)))
        return data, old_data, new_data, n_feature, n_class
        
    elif mode == 'graph':
        if dataset in ['PROTEINS_full', 'DBLP_v1']:
            data = TUDataset(root=path, name=dataset, use_node_attr=True, pre_transform=NormalizeFeatures())
        elif dataset in ['MUTAG', 'COLLAB', 'REDDIT-MULTI-12K', 'reddit_threads', 'REDDIT-BINARY']:
            data = TUDataset(root=path, name=dataset, use_node_attr=True, pre_transform=Compose([LocalDegreeProfile(), NormalizeFeatures(), close_grad]))
        n_data = len(data)
        n_class = data.num_classes
        n_feature = data.num_node_features
        
        index = [i for i in range(n_data)]
        random.shuffle(index)
        split = round((1-alpha) * n_data)
        old_data = index[:split]
        new_data = index[split:]
        print('old data {}, new data {}'.format(split, n_data-split))
        return data, data[old_data], data[new_data], n_feature, n_class
    
    return data
