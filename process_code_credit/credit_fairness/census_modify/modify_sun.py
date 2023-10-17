# 修改利用原网络结构census.pt导入sunbing的网络结构信息，得到census_sun.pt
import torch
import utils
import copy
import ast
import numpy as np

params = [
    (1, 64),
    (2, 32),
    (3, 16),
    (4, 8),
    (5, 4),
    (6, 2)
]

def modify(model, params):

    input = open('census_fairness/census_modify/model.txt', 'r')
    lines = input.readlines()
    
    print(len(lines))
    
    for i in range(6):
        wline = 1 + 2 * i
        bline = 2 + 2 * i
    
        w = np.array(ast.literal_eval(lines[wline]))
        b = np.array(ast.literal_eval(lines[bline]))
        w = w.transpose(1,0)
        w = torch.from_numpy(w)
        b = torch.from_numpy(b)
        # print(w)

        index = [64,32,16,8,4,2]

        layer = i + 1
        key1 = f'fc{layer}.weight'
        key2 = f'fc{layer}.bias'
        weight = model[key1]
        bias = model[key2]

        # print("before")
        # print(weight[0])
        # print(w[0])
        # print(bias[0])
        # print(b[0])

        for i in range(index[i]):
            weight[i] = w[i]
            bias[i] = b[i]
        print("after")
        print(weight)
        # print(bias[0])
        model[key1] = weight
        model[key2] = bias



    # for  param in params:
    #     layer, pos = param[0], param[1]
    #     key1 = f'fc{layer}.weight'
    #     key2 = f'fc{layer}.bias'

    #     weight = model[key1]
    #     bias = model[key2]
    #     for i in range(pos):
    #         weight[i] = f'w{i}'[i]
    #     model[key1] = weight
        
def main():
    file_name = "data_13/census.pt"
    base_model = utils.read_pt_file(file_name)

    model = copy.deepcopy(base_model) 
    modify(model, params, )
    utils.save(model, 'census_sun')

if __name__ == "__main__":
    main()
