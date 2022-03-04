# DarkNet Classifier Ver.3 (quantized Resnet18 CIFAR10, Lenet5 Mnist) 
fixed point nomalization is done in python

## DarkNet command

### lenet5
./darknet classifier predict cfg/mnist.data cfg/mnist_lenet.cfg backup/merged_weights_le5_mnist_normalized data/mnist_images/test/0_Five.png

### resnet18
./darknet classifier predict cfg/cifar10.data cfg/quantized_resnet18.cfg backup/merged_weights_resnet18_edited_normalized data/CIFAR10_images/bird/0100.jpg


## Avaiable operation

2dconvolution
Relu
maxpool
connected
shortcut
softmax

# python

## le5
le5 mnist model is referenced from https://velog.io/@jaewonalive/MNIST-Quantization-Aware-Training-example

lenet5_qat_mnist:Resnet18_qat_cifar10 :training model and save

lenet5_make scale_fixed_point: (from saved lenet5) fixed point normalization to ccale and save weights file to merged_weights and can see layer scales, zeropoints 

## resnet18
resnet18 cifar10 model is referenced from https://gaussian37.github.io/dl-pytorch-quantization/

Resnet18_qat_cifar10 :training model and save

resnet18_make scale_fixed_point: (from saved resnet18) fixed point normalization to scale and save weights file to merged_weights and can see layer scales, zeropoints 

resnet18_edit.ipynb: edit quantized resnet18 zeropoint and scale 


