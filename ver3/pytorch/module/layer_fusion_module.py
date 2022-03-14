##############fusefunction
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
CONV_LAYER= 'Conv2d'
BATCHNORM_LAYER= 'BatchNorm2d'
RELU_LAYER = 'ReLU'
def fusion_conv_bn_relu(layer):

    layer_list=dict()
    fusion_list=list()
    for basic_block_name, basic_block in layer.named_children():
        layer_list[basic_block_name]=basic_block._get_name()
    fusion_group=list()
    for layer_num in layer_list:
        if layer_list[layer_num] == CONV_LAYER:
            if fusion_group:
                fusion_list.append(fusion_group)
            fusion_group=list()
            fusion_group.append(layer_num)
        if layer_list[layer_num] == BATCHNORM_LAYER:
            if fusion_group:
                fusion_group.append(layer_num)
        if layer_list[layer_num] == RELU_LAYER:
            if fusion_group:
                fusion_group.append(layer_num)
    if fusion_group:
        fusion_list.append(fusion_group)
            
    return fusion_list        
            
def make_fuse_dict(model,model_name):

    fusion_layer_dict=dict()
    for module_name, module in model.named_children():
        class_path=model_name
        if str.isdigit(module_name): 
            class_path+='['+module_name+']'
        else:
            class_path+='.'+module_name
        fusion_list=list()
        fusion_list=fusion_conv_bn_relu(module)
        if fusion_list:        
            fusion_layer_dict[class_path]=fusion_list
        
        for module_name_, module_ in module.named_children():
            if str.isdigit(module_name_): 
                class_path_=class_path+'['+module_name_+']'
            else:
                class_path_=class_path+'.'+module_name_
            fusion_list=list()
            fusion_list=fusion_conv_bn_relu(module_)
            if fusion_list:        
                fusion_layer_dict[class_path_]=fusion_list
            for module_name__, module__ in module_.named_children():
                if str.isdigit(module_name__): 
                    class_path__=class_path_+'['+module_name__+']'
                else:
                    class_path__=class_path_+'.'+module_name__
                fusion_list=list()
                fusion_list=fusion_conv_bn_relu(module__)
                if fusion_list:        
                    fusion_layer_dict[class_path__]=fusion_list
                for module_name___, module___ in module__.named_children():
                    if str.isdigit(module_name___): 
                        class_path___=class_path__+'['+module_name___+']'
                    else:
                        class_path___=class_path__+'.'+module_name___
                    fusion_list=list()
                    fusion_list=fusion_conv_bn_relu(module___)
                    if fusion_list:        
                        fusion_layer_dict[class_path___]=fusion_list    
    return fusion_layer_dict

def fuse_model(model):
    fusion_layer_dict= make_fuse_dict(model,'model')
    print(fusion_layer_dict)
    for key in fusion_layer_dict:
        for fuse_group in fusion_layer_dict[key]:        
            torch.quantization.fuse_modules(eval(key), [fuse_group], inplace=True) 