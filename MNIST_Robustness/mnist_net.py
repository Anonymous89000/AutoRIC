# init robustness 1.25
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import onnx

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
train_losses = []
train_counter = []
test_losses = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = [
            
        ]
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(network, epoch, optimizer):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx /
                                                                           len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) +
                                 ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model_.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def data_down():

    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    # print(example_targets)
    # print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def cal_rob_radius(model = './model_.pth'):
    pos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    neg = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    p = 0 # + robustness
    n = 0 # - robustness

    if(isinstance(model, str)):
        network = Net()
        network.load_state_dict(torch.load(model))
    else:
        network = model

    for i in pos:
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)),
                                           torchvision.transforms.Lambda(
                                               lambda x: x + 0.25*i)
                                       ])),
            batch_size=batch_size_test, shuffle=True)

        acc = test(network, test_loader)
        if (acc < 0.0):
            p = i * 0.25
            break
        
    print('\nPositive radius : {:.2f}\n'.format(p))

    for j in neg:
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)),
                                           torchvision.transforms.Lambda(
                                               lambda x: x + 0.25*j)
                                       ])),
            batch_size=batch_size_test, shuffle=True)
        acc = test(network, test_loader)
        if (acc < 0.0):
            n = j * 0.25
            break

    print('\nNegitive radius : {:.2f}\n'.format(n))
    rob = p
    if (abs(n) < p):
        rob = abs(n)
    print('\nRobustness radius : {:.2f}\n'.format(rob))

    return rob

"""
def model_to_onnx(model):
    onnx_file = 'model.onnx'
    dummy_input = torch.randn(1,1, 28, 28, device="cpu")
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)
"""

if __name__ == "__main__":
    """
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    train(net, 10, optimizer)
    """

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

    network = Net()
    network.load_state_dict(torch.load('./'))
    test(network, test_loader)
    # data_down()
    # train(1)
    #network = Net()
    #print(network)
    #network.load_state_dict(torch.load('./model.pth'))
    # model_to_onnx(network)
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    # for epoch in range(0, n_epochs + 1):
    #     train(epoch)
    #     test()
