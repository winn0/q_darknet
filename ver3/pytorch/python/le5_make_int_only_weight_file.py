###########fusionx load saved_model
import sys
import time
import random
import os
sys.path.append("..")
from module.lenet5_module import *
from module.extract_weight_module import *
model_dir = os.getcwd()+'/saved_models'
model_filename='le5_saved.pt'
model_filepath=model_dir+'/'+model_filename

transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
    ])

device = 'cuda'

dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform = transform)
dataset2 = datasets.MNIST('../data', train=False, download=True,
                              transform = transform)


train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1)


start = time.time()

model_fp32 = Net()
model_fp32.train() # 아래에 진행될 Quantization Aware Training logic이 작동하기 위해서는 모델을 train 모드로 바꿔줘야 한다고 한다.
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
#model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'relu']])
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)
model_fp32_prepared = model_fp32_prepared.to("cuda")
model_fp32_prepared = load_model(model=model_fp32_prepared, model_filepath=model_filepath, device='cpu')

model_fp32_prepared.eval()
model_int8_unfused = torch.quantization.convert(model_fp32_prepared.to('cpu')) #quantized aware training을 floating point로 수행한 model을 quantized integer model로 바꿔준다.



model_int8_unfused.eval()


test_loss = 0
correct = 0

start2 = time.time()   
count =0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to('cpu'), target.to('cpu') #GPU는 integer형 연산을 지원하지 않으므로 추론 속도를 비교하기 위해서 모델과 data를 모두 cpu로 옮겨줬다.
        output = model_int8_unfused(data)
        input_data =data
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)


print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)
))

end = time.time()


#print("test 이전까지 경과 시간(secs):",start2-start)
print("inference를 할 때 걸린 시간(secs):",end-start2)
#print("total time elapsed(secs):", (end-start))

#####scale normalize and extract weights
extract_model=model_int8_unfused

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
merge_weight_files(save_list,"le5_mnist_weights_int_only")
