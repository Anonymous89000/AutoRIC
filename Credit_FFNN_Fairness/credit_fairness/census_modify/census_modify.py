import torch
import utils
import copy

params = [
    (1, 1),
    (1, 24),
    (1, 33),
    (1, 45),
    (2, 7),
    (2, 11),
    (2,18),
    (2, 25),
    (3, 7),
    (3, 11),
    (3, 12),
    (4, 3),
    (5, 3)
]

disturbs = [0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]

def modify(model, params, disturb):

    for  param in params:
        layer, pos = param[0], param[1] - 1
        key = f'fc{layer}.weight'

        weight = model[key]
        weight[pos] += disturb
        model[key] = weight
        
def main():
    file_name = "gender_fairness/census_modify/census.pt"
    base_model = utils.read_pt_file(file_name)
    
    for disturb in disturbs:
        model = copy.deepcopy(base_model) 
        modify(model, params, disturb)
        utils.save(model, disturb)

if __name__ == "__main__":
    main()