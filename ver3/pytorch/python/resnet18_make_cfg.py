#####load quanztized_resnet18
import sys
import time
import random
import os
sys.path.append("..")
from module.layer_fusion_module import *
from module.resnet_module import *
from module.extract_weight_module import *
from module.make_cfg_module import *
model_dir = "saved_models"
model_filename = "q_resnet18_cifar10.pt"
quantized_model_edited_filename = "q_resnet18_quantized_cifar10_edited.pt"
fusioned_model_filename = "q_fusioned_resnet18_cifar10.pt"
quantized_model_filename_jit = "q_resnet18_quantized_cifar10.pt"
quantized_model_filename = "q_resnet18_quantized_cifar10.pt"

model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
quantized_model_filepath_jit = os.path.join(model_dir, quantized_model_filename_jit)
quantized_model_edited_filepath = os.path.join(model_dir, quantized_model_edited_filename)

num_classes = 10
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

model = create_model(num_classes=num_classes)
model.to(cpu_device)

quantized_model = QuantizedResNet18(model_fp32=model)

quantization_config = torch.quantization.get_default_qconfig("fbgemm")

quantized_model.qconfig = quantization_config
fuse_model(quantized_model)
torch.quantization.prepare_qat(quantized_model, inplace=True)
quantized_model.to(cpu_device)    
quantized_model = torch.quantization.convert(quantized_model, inplace=True)
#save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename_jit)
#save_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)
loaded_model=load_model(model=quantized_model, model_filepath=quantized_model_edited_filepath, device='cpu')
loaded_model.eval()
#save each layer weights
import sys
import time
import random
import os
sys.path.append("..")
from module.make_cfg_module import *
input_width=32
input_height=32
input_channel=3
mean_list=list()
mean_list.append(0.485)
mean_list.append(0.456)
mean_list.append(0.406)
std_list=list()
std_list.append(0.229)
std_list.append(0.224)
std_list.append(0.225)
make_cfg_from_pytorch(quantized_model,"resnet18_cifar10.cfg",input_width,input_height,input_channel,mean_list,std_list)