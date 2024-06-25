import torch


def read_pt_file(file):
    return torch.load(file)

def save(model, disturb):
    save_file = f'census_fairness/census_modify/{disturb}.pt'
    torch.save(model, save_file)

if __name__ == "__main__":
    # model = torch.load("-0.1.pt")
    # print(model['fc3.weight'][10])
    # base = torch.load("census.pt")
    # print(base['fc3.weight'][10])
    print("sucess")