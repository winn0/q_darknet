###save function def
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
weight_dir ='weight_data/'
def save_np_1darray(save_list,file_name,save_np_file):
    
    if (save_np_file.dtype =='int8') or (save_np_file.dtype =='uint8'):
        save_list[file_name]=save_np_file.size
    elif (save_np_file.dtype =='float16'):
        save_list[file_name]=save_np_file.size*2
    elif (save_np_file.dtype =='float32') or (save_np_file.dtype =='int32') or (save_np_file.dtype =='uint32') : 
        save_list[file_name]=save_np_file.size*4
    elif (save_np_file.dtype =='float64'):
        save_list[file_name]=save_np_file.size*4
        save_np_file=save_np_file.astype(np.float32)
    elif (save_np_file.dtype =='int64'):
        save_list[file_name]=save_np_file.size*4
        save_np_file=save_np_file.astype(np.int32)
    else:
        assert 0,"unknown file type error"

    save_np_file.tofile(weight_dir+file_name)  
    return save_list
def make_default_save_file(save_list):
    file_name ='default_setting'
    default_setting = np.zeros((16,), dtype=np.int8)
    default_setting[0] = 0
    default_setting[1] = 0
    default_setting[2] = 0
    default_setting[3] = 0    
    default_setting[4] = 1
    default_setting[5] = 0
    default_setting[6] = 0
    default_setting[7] = 0
    default_setting[8] = 0
    default_setting[9] = 0
    default_setting[10] = 0
    default_setting[11] = 0
    default_setting[12] = 0
    default_setting[13] = 0
    default_setting[14] = 0
    default_setting[15] = 0
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    default_setting.tofile(weight_dir+file_name)    
    save_list[file_name]=16
    return save_list

def merge_weight_files(save_list):
    sum_size=0
    for file_name in save_list:
        sum_size += save_list[file_name]
    merged_weights = np.zeros((sum_size,), dtype=np.int8)
    save_checkpoint=0
    for file_name in save_list:
        save_end = save_checkpoint+ save_list[file_name]
        temp_weights=np.fromfile(weight_dir+file_name,dtype=np.int8)
        #print(temp_weights)
        merged_weights[save_checkpoint:save_end]=temp_weights
        save_checkpoint=save_end
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    merged_weights.tofile(weight_dir+"merged_weights")
        
    