import torch
import random
import numpy as np

from util.lib_models import *
# from lib_layers import *

# from poly_utils import *
# from refinement_impl import Poly

from util.utils import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# import gurobipy as gp
# from gurobipy import GRB

import math
import ast

# from nn_utils import *

import time


class CreditNet(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.fc1 = nn.Linear(num_of_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = x # cross entropy in pytorch already includes softmax
        return output

def __get_data(input_file, label_file):                     
    input_file = open(input_file, 'r'  )                            
    input_data = []                                               
    for line in input_file.readlines():                           
        input_data.append(ast.literal_eval(line))                 
    input_data = np.array(input_data)                             
                                                                  
    label_file = open(label_file, 'r')                            
    label_data = np.array(ast.literal_eval(label_file.readline()))
                                                                  
    return input_data, label_data       

def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # print(x.type)

        # Compute prediction error
        pred = model(x)
        # print(pred)
        # print(y)
        pred = pred.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0: ###########################################
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")                          
 
def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # print(size)
    # print(num_batches)
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            pred = pred.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            # print(pred)
            # print(y)
            test_loss += loss_fn(pred, y).item()
            # print(pred.argmax(1))
            # print(y.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).sum().item()
    
    test_loss /= num_batches
    error = size - correct
    correct /= size

    print(f"Test: \n Test size: {(size):>0.1f}, Error size: {(error):>0.1f}, Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct

def save_model(model, name):
    torch.save(model.state_dict(), name)

def __train_model(model, train_x, train_y, test_x, test_y, device, file_name, fair_x=None, fair_y=None, lst_poly_lst=None):   
    tensor_train_x = torch.FloatTensor(train_x.copy())   # transform to torch tensor                                                       
    tensor_train_y = torch.FloatTensor(train_y.copy())                                                           
                                                                                                                                    
    tensor_test_x = torch.FloatTensor(test_x.copy())                                                            
    tensor_test_y = torch.FloatTensor(test_y.copy())                                                              
                                                                                                                                    
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset                                                  
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True) # create dataloader                                  
                                                                                                                                    
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=10) # create dataloader                                                  
                                                                                                                                    
    optimizer = optim.SGD(model.parameters(), lr=0.01)############################
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   
    criterion = nn.CrossEntropyLoss()       
    # 定义训练批次                                                                    
    num_of_epochs = 100#################################                                                                                                              
                                                                                                                                    
    if fair_x is not None:                                                                                                          
        tensor_fair_x = torch.Tensor(fair_x.copy()) # transform to torch tensor                                                     
        tensor_fair_y = torch.Tensor(fair_y.copy()).type(torch.FloatTensor)                                                          
                                                                                                                                    
        fair_dataset = TensorDataset(tensor_fair_x, tensor_fair_y) # create dataset                                                 
        fair_dataloader = DataLoader(fair_dataset, batch_size=10, shuffle=False) # create dataloader                                
                                                                                                                                    
    best_acc = 0.0                                                                                                                  
                                                                                                                                    
    start = time.time()                                                                                                             
    for epoch in range(num_of_epochs):                                                                                              
        print('\n------------- Epoch {} -------------\n'.format(epoch))                                                             
        if fair_x is None:                                                                                                          
            # print('Use normal loss funtion!!!')                                                                                     
            train(model, train_dataloader, criterion, optimizer, device)                                                
        # else:                                                                                                                       
        #     print('Use special loss funtion!!!')                                                                                    
        #     __train_fair(model, train_dataloader, fair_dataloader, nn.CrossEntropyLoss(), optimizer, device, lst_poly_lst)     
        test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), device)                                                      

        # 将训练好的机器学习模型保存到指定文件路径中                                                                                                                           
        if best_acc < test_acc:                                                                                                     
            best_acc = test_acc                                                                                                     
            save_model(model, file_name)                                                                                            
    end = time.time()                                                                                                               
                                                                                                                                    
    return end - start                                                                                                              

def run():
    lamdba = 1e-3

    device = 'cpu'                    
    train_kwargs = {'batch_size': 100}
    test_kwargs = {'batch_size': 1000}

    # 定义特征数! ! !
    num_of_features =  20                       
    gender_idx = 8                               

    # 初始化模型                                             
    model = CreditNet(num_of_features).to(device)
    
    # 读取输入数据和标签
    train_x = np.loadtxt('data/data_numeric/trainx.txt')
    train_y = np.loadtxt('data/data_numeric/trainy.txt')
    test_x = np.loadtxt('data/data_numeric/testx.txt')
    test_y = np.loadtxt('data/data_numeric/testy.txt')
    print(f"The number of training:  {train_x[:,].shape} \n")
    print(f"The number of testing:  {test_x[:,].shape} \n")

    train_kwargs = {'batch_size': 100}

    # 训练模型
    time = __train_model(model,train_x,train_y,test_x,test_y,device,'data/data_numeric/credit.pt')
    print(f"The time of training:  {time:>0.5f} \n")

if __name__ == '__main__':
    run()
# for batch, (x, y) in enumerate(train_dataloader):     
#     x, y = x.to(device), y.to(device)                 
                                                      
#     # Compute prediction error                        
#     pred = model(x)                                   
#     loss = criterion(pred, y)                           
                                                      
#     # Backpropagation                                 
#     optimizer.zero_grad()                             
#     loss.backward()                                   
#     optimizer.step()                                  
                                                      
# model.fc1.register_forward_hook(get_activation('fc1'))
# model.fc2.register_forward_hook(get_activation('fc2'))
# model.fc3.register_forward_hook(get_activation('fc3'))
# model.fc4.register_forward_hook(get_activation('fc4'))
# model.fc5.register_forward_hook(get_activation('fc5'))
# model.fc6.register_forward_hook(get_activation('fc6'))

# for batch, (x, y) in enumerate(fair_dataloader):                                                         
#     x, y = x.to(device), y.to(device)                                                                    
#     lst_poly = lst_poly_lst[batch]                                                                       
                                                                                                         
#     # Compute prediction error                                                                           
#     pred = model(x)                                                                                      
#     loss = 0.0                                                                                           
                                                                                                         
#     for i in range(6):                                                                                   
#         layer = 'fc' + str(i + 1)                                                                        
#         layer_tensor = activation[layer]                                                                 
                                                                                                         
#         lower_tensor = torch.Tensor(lst_poly[2 * i + 1].lw)                                              
#         upper_tensor = torch.Tensor(lst_poly[2 * i + 1].up)                                              
#         mean_tensor = (lower_tensor + upper_tensor) / 2                                                  
                                                                                                         
#         mask_lower = layer_tensor < lower_tensor                                                         
#         mask_upper = layer_tensor > upper_tensor                                                         
                                                                                                         
#         # square                                                                                         
#         sum_lower = ((layer_tensor - mean_tensor) ** 2)[mask_lower].sum()                                
#         sum_upper = ((layer_tensor - mean_tensor) ** 2)[mask_upper].sum()                                
                                                                                                         
#         # all layers                                                                                     
#         loss = loss + lamdba * (sum_lower + sum_upper) / (len(layer_tensor) * len(layer_tensor[0]))      
                                                                                                         
#     # Backpropagation                                                                                    
#     optimizer.zero_grad()                                                                                
#     loss.backward()                                                                                      
#     optimizer.step()                                                                                     
