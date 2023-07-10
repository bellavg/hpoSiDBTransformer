from hyperparameters import GRIDSIZE
from block import Block
from positionalinformation import *
from customloss import *
import torch
import torch.nn as nn
import MinkowskiEngine as ME



class SiDBTransformer(nn.Module):
    def __init__(self,  embeddim, position_info, depth, heads,
                  d_rate, input_dim=1, gridsize=GRIDSIZE, num_classes=2):
        super().__init__()

        self.pe_type = position_info

        self.tblocks = nn.ModuleList(
            [Block(h=heads, pe=position_info, ed=embeddim, d_rate=d_rate) for _ in range(depth)])

        self.to_probs = ME.MinkowskiLinear(embeddim, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

        self.medrop = ME.MinkowskiDropout(d_rate)

        self.norm = nn.BatchNorm2d(num_features=num_classes)

        self.drop = nn.Dropout(d_rate)

        self.to_linprobs = nn.Linear(embeddim, num_classes)

        self.embedconv = ME.MinkowskiConvolution(in_channels=input_dim, out_channels=embeddim, bias=True, kernel_size=3, dimension=2)


        self.embed = ME.MinkowskiLinear(in_features=input_dim, out_features=embeddim, bias=True)

        if self.pe_type == "base":
            self.get_pos = BaseAbsPE(embeddim,  gridsize)


    def forward(self, x):

        b, gs, _ = x.shape

        nzmask = x.view(-1) > 0
        nonzero_coords = torch.nonzero(x).contiguous().int()

        if self.pe_type == "physical2":
            dis_pos = get_physical(x.clone().detach(), mindim=True)
            x = x + dis_pos.squeeze(-1).to(x.device)
            x = self.drop(x)
            del dis_pos
        elif self.pe_type == "potential":
            elpot = get_potential(x.clone().detach(), b, gs)
            x += elpot
            x = self.drop(x)
            del elpot

        x = x.reshape(-1)
        x = x[nzmask]
        x = x.unsqueeze(-1)

        quantization_mode = ME.SparseTensorQuantizationMode.NO_QUANTIZATION
        x = ME.SparseTensor(coordinates=nonzero_coords.to(x.device), features=x,  quantization_mode=quantization_mode)
        del nonzero_coords


        x = self.embed(x)

        nzmask = nzmask.unsqueeze(-1).repeat(1, x.shape[-1])

        if self.pe_type == "AN":
            pos = positionalencoding2d(x.shape[-1], gs, gs, b)  # returns cardinal coordinates embedded or sin/cosine
            pos = pos.reshape(-1, x.shape[-1]).to(x.device)
            pos = pos[nzmask].view(x.shape)
            pos = ME.SparseTensor(features=pos.to(x.device), coordinates=x.coordinates,
                                  coordinate_manager=x.coordinate_manager, quantization_mode=quantization_mode)
            x = x + pos
            x = self.medrop(x)
            del pos

        if self.pe_type == "base":
            pos = self.get_pos(b)  # returns cardinal coordinates embedded or sin/cosine
            pos = pos.reshape(-1, x.shape[-1]).to(x.device)
            pos = pos[nzmask].view(x.shape)
            pos = ME.SparseTensor(features=pos.to(x.device), coordinates=x.coordinates,
                                  coordinate_manager=x.coordinate_manager, quantization_mode=quantization_mode)
            x = x + pos
            x = self.medrop(x)
            del pos

        for blk in self.tblocks:
            x = blk(x, nzmask, b, gs)

        x = self.to_probs(x)

        x,_,_ = x.dense(shape=torch.Size([b,  2, gs, gs])) #check size

        X = self.norm(x)


        #x = self.to_linprobs(x) #check #check size

        x = self.sig(x) #check size


        return x #check size
