#####training q_resnet18
import sys
import time
import random
import os
sys.path.append("..")
from module.layer_fusion_module import *
from module.resnet_module import *
random_seed = 0
num_classes = 10
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

model_dir = "saved_models"
model_filename = "q_resnet18_cifar10.pt"
quantized_model_filename = "q_resnet18_quantized_cifar10.pt"
model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

set_random_seeds(random_seed=random_seed)
    
    # Create an untrained model.
model = create_model(num_classes=num_classes)

train_loader, test_loader = prepare_dataloader(num_workers=0, train_batch_size=128, eval_batch_size=256)
    
# Train model.
print("Training Model...")
model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-1, num_epochs=10)
# Save model.
save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    
# ① floating point 타입으로 모델을 학습하거나 pre-trained 모델을 불러옵니다.
# Load a pretrained model.
model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
# Move the model to CPU since static quantization does not support CUDA currently.
    
# ② 모델을 CPU 상태로 두고 학습 모드로 변환합니다. (model.train())
model.to(cpu_device)
# Make a copy of the model for layer fusion
fused_model = copy.deepcopy(model)

model.train()
# The model has to be switched to training mode before any layer fusion.
# Otherwise the quantization aware training will not work correctly.
fused_model.train()

#QAT
def fuse_model(model,model_name):
    fusion_layer_dict= make_fuse_dict(model,model_name)
    for key in fusion_layer_dict:
        for fuse_group in fusion_layer_dict[key]:        
            torch.quantization.fuse_modules(eval(key), [fuse_group], inplace=True) 
model_dir = "saved_models"
model_filename = "q_resnet18_cifar10.pt"
fusioned_model_filename = "q_fusioned_resnet18_cifar10.pt"
quantized_model_filename_jit = "q_resnet18_quantized_cifar10_jit.pt"
quantized_model_filename = "q_resnet18_quantized_cifar10.pt"
model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
quantized_model_filepath_jit = os.path.join(model_dir, quantized_model_filename_jit)

quantized_model = QuantizedResNet18(model_fp32=fused_model)

quantization_config = torch.quantization.get_default_qconfig("fbgemm")

quantized_model.qconfig = quantization_config
fuse_model(quantized_model,'quantized_model')
torch.quantization.prepare_qat(quantized_model, inplace=True)

print("Training QAT Model...")
quantized_model.train()

train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-3, num_epochs=10)
quantized_model.to(cpu_device)    
quantized_model = torch.quantization.convert(quantized_model, inplace=True)

quantized_model.eval()
save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename_jit)
save_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

# Load quantized model_jit.
quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath_jit, device=cpu_device)
_, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cuda_device, criterion=None)
_, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)
#_, int8_eval_accuracy = evaluate_model(model=quantized_model, test_loader=test_loader, device=cpu_device, criterion=None)
print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)

print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))