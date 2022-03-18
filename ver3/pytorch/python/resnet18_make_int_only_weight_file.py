#####load quanztized_resnet18
import sys
import time
import random
import os
sys.path.append("..")
from module.layer_fusion_module import *
from module.resnet_module import *
from module.extract_weight_module import *
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


extract_model=loaded_model

SCALE='scale'
WEIGHT='weight'
BIAS='bias'
FC='_packed_params'
save_list= dict()
save_list=make_default_save_file(save_list)
scale_list=list()
weight_input_scale_index_dict=dict()
weight_dict=dict()
weight_channel_scale_dict=dict()
weight_channel_zeropoint_dict=dict()
weight_channel_bias_dict=dict()
scale_count=-1
for parameter_name in extract_model.state_dict():
    print(parameter_name)
    if parameter_name[-len(SCALE):]==SCALE:
        scale_count+=1
        scale_list.append(extract_model.state_dict()[parameter_name].numpy())
    if parameter_name[-len(WEIGHT):]==WEIGHT:
        file_name=parameter_name[:-len(WEIGHT)-1]
        weight_input_scale_index_dict[file_name]=scale_count
        weight_dict[file_name]=extract_model.state_dict()[parameter_name].int_repr().numpy().flatten()
        weight_channel_scale_dict[file_name]=extract_model.state_dict()[parameter_name].q_per_channel_scales().numpy()
        weight_channel_zeropoint_dict[file_name]=extract_model.state_dict()[parameter_name].q_per_channel_zero_points().numpy()
    if parameter_name[-len(BIAS):]==BIAS:
        weight_channel_bias_dict[file_name]=extract_model.state_dict()[parameter_name].detach().numpy()
    if parameter_name[-len(FC):]==FC:
        file_name=parameter_name[:-len(FC)-1]
        weight_input_scale_index_dict[file_name]=scale_count-1 #fc layer scale is faster than weight
        weight_dict[file_name]=extract_model.state_dict()[parameter_name][0].int_repr().numpy().flatten()
        weight_channel_scale_dict[file_name]=extract_model.state_dict()[parameter_name][0].q_per_channel_scales().numpy()
        weight_channel_zeropoint_dict[file_name]=extract_model.state_dict()[parameter_name][0].q_per_channel_zero_points().numpy()
        weight_channel_bias_dict[file_name]=extract_model.state_dict()[parameter_name][1].detach().numpy()
        
for parameter_name in extract_model.state_dict():
    print(parameter_name)
    if parameter_name[-len(WEIGHT):]==WEIGHT:
        file_name=parameter_name[:-len(WEIGHT)-1]
        M_in32_np, right_shift_np, bias_int32_np =make_channel_nomalization(scale_list[weight_input_scale_index_dict[file_name]], weight_channel_scale_dict[file_name],
                                 scale_list[weight_input_scale_index_dict[file_name]+1],weight_channel_bias_dict[file_name])
        save_np_file=weight_dict[file_name]
        save_list=save_np_1darray(save_list,file_name+"_weight",save_np_file)
        save_np_file=M_in32_np
        save_list=save_np_1darray(save_list,file_name+"_M0",save_np_file)
        save_np_file=right_shift_np
        save_list=save_np_1darray(save_list,file_name+"_rightshift",save_np_file)
        save_np_file=weight_channel_zeropoint_dict[file_name]
        save_list=save_np_1darray(save_list,file_name+"_zeropoint",save_np_file)
        save_np_file=bias_int32_np
        save_list=save_np_1darray(save_list,file_name+"_bias",save_np_file)
    if parameter_name[-len(FC):]==FC:
        file_name=parameter_name[:-len(FC)-1]
        M_in32_np, right_shift_np, bias_int32_np =make_channel_nomalization(scale_list[weight_input_scale_index_dict[file_name]], weight_channel_scale_dict[file_name],
                                 scale_list[weight_input_scale_index_dict[file_name]+1],weight_channel_bias_dict[file_name])
        save_np_file=weight_dict[file_name]
        save_list=save_np_1darray(save_list,file_name+"_weight",save_np_file)
        save_np_file=M_in32_np
        save_list=save_np_1darray(save_list,file_name+"_M0",save_np_file)
        save_np_file=right_shift_np
        save_list=save_np_1darray(save_list,file_name+"_rightshift",save_np_file)
        save_np_file=weight_channel_zeropoint_dict[file_name]
        save_list=save_np_1darray(save_list,file_name+"_zeropoint",save_np_file)
        save_np_file=bias_int32_np
        save_list=save_np_1darray(save_list,file_name+"_bias",save_np_file)


#saved layer info
for name in save_list:
    print(name, "  data size:",save_list[name], "byte")
#merge weights
merge_weight_files(save_list,"resnet18_cifar10_weights_int_only")