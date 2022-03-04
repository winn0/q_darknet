import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import time
import os

# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     self.quant = torch.quantization.QuantStub() 
#     self.conv1 = nn.Conv2d(1, 32, 3, 1)
#     self.conv2 = nn.Conv2d(32, 64, 3, 1)
#     self.relu = torch.nn.ReLU()
#     self.dropout1= nn.Dropout(0.25)
#     self.dropout2 = nn.Dropout(0.5)
#     self.fc1 = nn.Linear(9216, 128)
#     self.fc2 = nn.Linear(128, 10)
#     self.maxpool = nn.MaxPool2d(2)
#     self.logsoftmax = torch.nn.LogSoftmax(dim=1)
#     self.dequant = torch.quantization.DeQuantStub()

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.quant = torch.quantization.QuantStub() 
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.relu = torch.nn.ReLU()
    self.maxpool = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.relu2 = torch.nn.ReLU()
    self.fc1 = nn.Linear(7744, 128)
    self.relu3 = torch.nn.ReLU()
    self.fc2 = nn.Linear(128, 10)
    self.relu4 = torch.nn.ReLU()
    self.dequant = torch.quantization.DeQuantStub()
    self.logsoftmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.quant(x) # QuanStub를 forward를 시작하는 부분에 적어준다.
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.relu2(x)
#   x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.relu3(x)
    x = self.fc2(x)
    x = self.relu4(x)
    x = self.dequant(x) # DeQuanStub를 forward가 끝나는 부분에 적어준다. LogSoftmax의 경우, 후에 추론 시에 사용할 데이터 형태인 QuantizedCPU형을 지원하지 않으므로, LogSofmax이전에 dequantization을 해준다.
    x = self.logsoftmax(x)
    
    return x
def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model