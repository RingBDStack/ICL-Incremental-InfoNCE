import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from conv import MetaGCNConv
import numpy as np
from collections import OrderedDict
from resnet import Bottleneck, BasicBlock
import math
from torchmeta.modules import DataParallel, MetaModule, MetaConv2d, MetaBatchNorm2d, MetaLinear, MetaSequential



class GraphEncoder(MetaModule):
    def __init__(self, encoder, in_fea, dim, layer_num):
        super(GraphEncoder, self).__init__()

        self.gnns = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.layer_num = layer_num
        self.dim = dim
        self.embedding_dim = dim * layer_num
        
        if encoder == 'gcn':
            for i in range(self.layer_num):
                if i:
                    gnn = MetaGCNConv(dim, dim)
                else:
                    gnn = MetaGCNConv(in_fea, dim)
            
                # nn.BatchNorm1d(dim),
                layer = nn.Sequential(nn.BatchNorm1d(dim, affine=False), nn.LeakyReLU(), nn.Dropout(p=0.5))
                # layer = nn.ReLU()
                self.gnns.append(gnn)
                self.layers.append(layer)
        
        # self.projection = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(), nn.Linear(self.embedding_dim, self.embedding_dim))
        
        
    def forward(self, data, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.layer_num):            
            x = self.gnns[i](x, edge_index, None, params['gnns.'+str(i)+'.weight'], params['gnns.'+str(i)+'.bias'])
            x = self.layers[i](x)
            xs.append(x)
            
        ret = torch.cat([global_add_pool(x, batch) for x in xs], 1)
        # x = self.projection(x)
        
        return ret
    
    def get_embeddings(self, loader, params=None):
        embs = []
        y = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))              
                emb = self.forward(data, params)
                embs.append(emb.cpu().numpy())
                y.append(data.y.cpu().numpy())
                
        embs = np.concatenate(embs, 0)
        y = np.concatenate(y, 0)
        return embs, y
    
    def get_dim(self):
        return self.embedding_dim
    
    
class ImgEncoder(MetaModule):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ImgEncoder, self).__init__()
        self.conv1 = MetaConv2d(3, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = MetaLinear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return MetaSequential(*layers)
        
        
    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        x = x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        x = self.bn1(x, params=self.get_subdict(params, 'bn1'))
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, params=self.get_subdict(params, 'layer1'))
        x = self.layer2(x, params=self.get_subdict(params, 'layer2'))
        x = self.layer3(x, params=self.get_subdict(params, 'layer3'))
        x = self.layer4(x, params=self.get_subdict(params, 'layer4'))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, params=self.get_subdict(params, 'fc'))

        x = nn.functional.normalize(x, dim=1, p=2)
        
        return x
    
    def get_embeddings(self, loader, params=None):
        embs = []
        y = []
        
        with torch.no_grad():
            for batch in loader:
                data = batch[0][0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                emb = self.forward(data, params)
                embs.append(emb.cpu().numpy())
                y.append(batch[1])
                
        embs = np.concatenate(embs, 0)
        y = np.concatenate(y, 0)
        return embs, y
    
    def get_dim(self):
        return 1000
        

def build_encoder(mode, encoder, in_fea, dim, layer_num):
    if mode == 'cv':
        if encoder == 'resnet18':
            model = ImgEncoder(BasicBlock, [2, 2, 2, 2])
        elif encoder == 'resnet34':
            model = ImgEncoder(BasicBlock, [3, 4, 6, 3])
        elif encoder == 'resnet50':
            model = ImgEncoder(Bottleneck, [3, 4, 6, 3])
        elif encoder == 'resnet101':
            model = ImgEncoder(Bottleneck, [3, 4, 23, 3])
        elif encoder == 'resnet18':
            model = ImgEncoder(Bottleneck, [3, 8, 36, 3])
            
    elif mode == 'graph':
        model = GraphEncoder(encoder, in_fea, dim, layer_num)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model

