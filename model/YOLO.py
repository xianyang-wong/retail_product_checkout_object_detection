import torch 
import torch.nn as nn
import numpy as np
from model.YOLOLayer import YOLOLayer
from model.EmptyLayer import EmptyLayer
from utils.utils import *
from collections import defaultdict


class YOLO(nn.Module):
    def __init__(self, cfgfile,num_classes):
        super(YOLO, self).__init__()
        self.blocks = parseModelConfig(cfgfile)
        self.num_classes = num_classes
        self.net_info, self.module_list = createModules(self.blocks)
        self.seen = 0
        self.header = torch.IntTensor([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
                
    def forward(self, x, CUDA, targets=None):
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        # first block is net info
        blocks = self.blocks[1:]

        for i, (block, module) in enumerate(zip(blocks, self.module_list)):
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif block["type"] == "route":
                layer_i = [int(x) for x in block["layers"]]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif block["type"] == "shortcut":
                layer_i = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif block["type"] == "yolo":
                inp_dim = int (self.net_info["height"])
                if targets is not None:
                    x, *losses = module[0](x, inp_dim, self.num_classes, CUDA, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                else:
                    x = module[0](x, inp_dim, self.num_classes, CUDA, targets=None)
                output.append(x)
            layer_outputs.append(x)

        # three yolo layers
        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if targets is not None else torch.cat(output, 1)

    def loadModel(self, weightfile):  
        #Open the weights file
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #number of weights of batch norm
                    num_bn_biases = bn.bias.numel()
                    
                    #load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    num_biases = conv.bias.numel()

                    #the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #copy the data
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
            
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def saveModel(self, savedfile, cutoff = 0):        
        fp = open(savedfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header = header.numpy()
        header.tofile(fp)
        
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0 
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    bn.bias.data.cpu().numpy().tofile(fp)
                    bn.weight.data.cpu().numpy().tofile(fp)
                    bn.running_mean.cpu().numpy().tofile(fp)
                    bn.running_var.cpu().numpy().tofile(fp)
                else:
                    conv.bias.data.cpu().numpy().tofile(fp)
                conv.weight.data.cpu().numpy().tofile(fp)

