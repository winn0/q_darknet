# DarkNet Classifier Ver.2 (quantized Resnet18 CIFAR10) 
fixed point nomalization is done in darknet

## DarkNet command

./darknet classifier predict cfg/cifar10.data cfg/quantized_resnet18.cfg backup/merged_weights_resnet18_edited data/CIFAR10_images

## Avaiable operation

2dconvolution
Relu
maxpool
connected
shortcut
softmax

# python
resnet18 cifar10 model is referenced from https://gaussian37.github.io/dl-pytorch-quantization/

Resnet18_qat_cifar10 :training model and save

extract_weight : save weights file to merged_weights and can see layer scales, zeropoints 

edit_resnet18.ipynb: edit quantized resnet18 zeropint and scale 
