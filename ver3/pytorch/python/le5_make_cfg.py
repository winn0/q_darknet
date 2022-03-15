###########load mnist saved model
import sys
import time
import random
import os
sys.path.append("..")
from module.lenet5_module import *
from module.extract_weight_module import *
from module.make_cfg_module import *
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
input_width=28
input_height=28
input_channel=1
mean_list=list()
mean_list.append(0.1307)
mean_list.append(0.1307)
mean_list.append(0.1307)
std_list=list()
std_list.append(0.3081)
std_list.append(0.3081)
std_list.append(0.3081)
make_cfg_from_pytorch(model_int8_unfused,"mnist_le.cfg",input_width,input_height,input_channel,mean_list,std_list)