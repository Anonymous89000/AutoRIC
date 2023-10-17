import argparse
import torch
import torchvision
import logging
from quadratic_fit import quadratic_fit
from deepZ import Conv
from deepZ.code import zonotope
from mnist_net import test
from verify import binarySearch
from cvx_optimize import cvx_optimize

torch.set_grad_enabled(True)

logging.basicConfig(level= 20, format="%(asctime)s :: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda"
INPUT_SIZE = 28
DELTA = 10e-4

# opt parameters
disturbs_0 = [0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]
disturbs_1 = [-0.001, 0.001]
#n*100 fc1
params_index_1=[
    (6, 1),
    (6, 24),
    (6, 43),
    (6, 65)
]

#n*100 fc2
params_index_2=[
    (8, 7),
    (8, 34),
    (8, 52),
    (8, 97)
]

#n*10 fc3
params_index_2=[
    (10, 1),
    (10, 3),
    (10, 5),
    (10, 7)
]

parser = argparse.ArgumentParser(description="Neural network verification using DeepZ relaxation")
parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
args = parser.parse_args()

def main():
    """
        optimize roubustness of conv4
    """
    l = DELTA
    r = 0.2

    # 0. read pixel data
    with open(args.spec, "r") as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        # eps = float(args.spec[:-4].split("/")[-1].split("_")[-1])

    # 1.establish the net work
    # we use network conv4
    network = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    print(network.layers)
    
    # 2.read the pt file
    pt_file = "./DeepZ/mnist_nets/conv4.pt"
    network.load_state_dict(torch.load(pt_file))
    # test acc

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=1000, shuffle=True)
    test(network, test_loader)

    # 3.binary search the MRR
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    max_robust_radius = binarySearch(network, inputs, true_label, l, r)
    print(f"MRR: {max_robust_radius}")

    # 4.optimize the network
    quadratic_fit(pt_file, 400, "mnist", params_index_1, disturbs_0, max_robust_radius, inputs, true_label, l, r)

    # 5.optimization
    opt_file = "result/mnist_opt.pt"
    cvx_optimize("mnist", params_index_1, 0)

    # 6.test acc and robust
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=1000, shuffle=True)
    network.load_state_dict(torch.load(opt_file))
    test(network, test_loader)
    max_robust_radius = binarySearch(network, inputs, true_label, l, r)
    print(f"optmized MRR: {max_robust_radius}")

if __name__ == "__main__":
    main()
    """
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=1000, shuffle=True)
    with open(args.spec, "r") as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
      
    opt_file = "result/mnist_opt.pt"
    network = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    network.load_state_dict(torch.load(opt_file))
    test(network, test_loader)
    l = DELTA
    r = 0.14
    max_robust_radius = binarySearch(network, inputs, true_label, l, r)
    print(f"optmized MRR: {max_robust_radius}")
    """