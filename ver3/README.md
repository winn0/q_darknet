# Quantized darkNet classifier Ver.3 (quantized Resnet18 CIFAR10, Lenet5 Mnist) 
fixed point nomalization is done in python
## Environment
This code was developed and testsed with
* ubuntu 18.04.6
* python 3.7.11 
* pytorch 1.10.2 
* CUDA 11.6
## DarkNet

### lenet5
#### setup 
1. cd darknet/data
2. python download_and_convert_mnist.py 
3. cd mnist_images and python make-mnist_list.py

#### classify
```
./darknet classifier predict cfg/mnist.data cfg/le5_mnist.cfg backup/le5_mnist_weights_int_only data/mnist_images/test/0_Five.png
```
#### validate
```
./darknet classifier valid cfg/mnist.data cfg/le5_mnist.cfg backup/le5_mnist_weights_int_only
```
### resnet18
##### setup
1. cd darknet/data
2. wget http://pjreddie.com/media/files/cifar.tgz
3. tar xzf cifar.tgz
4. python make_cifar10_list.py
#### classify
```
./darknet classifier predict cfg/cifar10.data cfg/resnet18_cifar10.cfg backup/resnet18_cifar10_weights_int_only data/cifar/test/0_cat.png
```
#### validate
```
./darknet classifier valid cfg/cifar10.data cfg/resnet18_cifar10.cfg backup/resnet18_cifar10_weights_int_only
```
## Avaiable operation
```
2dconvolution
Relu
maxpool
connected
shortcut
softmax
```
## pytorch

### le5

le5 mnist model is referenced from https://velog.io/@jaewonalive/MNIST-Quantization-Aware-Training-example

```
le5_qat_mnist:Resnet18_qat_cifar10 :training model and save

le5_make_int_only_weight_file: (from saved le5) fixed point normalization is applied to scale and save as weight_data/le5_mnist_weights_int_only

le5_make_combined_weight_file: (from saved le5) save both of fixed normzlized value and float value, save as weight_data/le5_mnist_weights_combined

le5_make_cfg:make cfg file for darknet from pytorch le5 model
```
### resnet18

resnet18 cifar10 model is referenced from https://gaussian37.github.io/dl-pytorch-quantization/

```
Resnet18_qat_cifar10 :training model and save

resnet18_make_int_only_weight_file: (from saved resnet18) fixed point normalization is applied to scale and save as weight_data/resnet18_cifar10_weights_int_only

resnet18_make_combined_weight_file: (from saved resnet18) save both of fixed normzlized value and float value, save as weight_data/resnet18_cifar10_weights_combined

resnet18_make_cfg:make cfg file for darknet from pytorch resnet18 model

resnet18_edit.ipynb: edit quantized resnet18 zeropoint and scale 
```


