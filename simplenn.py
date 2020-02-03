# Imports
import sys
import argparse
import os 
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_exp')

# Setup
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
        help='input batch size for training (default: 64)') 
parser.add_argument('--epochs', type=int, default=5, metavar='N',
        help='number of epochs to train (default:10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=True,
        help='resume training from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cudnn.enabled = False


# Data

data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
train_data = datasets.FashionMNIST(data_path, train=True, download=True,
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))]))
test_data = datasets.FashionMNIST(data_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size,
                         num_workers=1, pin_memory=True)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

################################################
# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig, ax = plt.subplots()
    if one_channel:
        ax.imshow(npimg, cmap="Greys")
    else:
        ax.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# create grid of images 
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
###################################################



# Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if args.resume:
    model.load_state_dict(torch.load('model.pth'))
    optimiser.load_state_dict(torch.load('optimiser.pth'))

#################################################
# Tenworboard Graph and Embedding
writer.add_graph(model, images)

# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(train_data.data, train_data.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))

writer.close()

# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
#################################################


# Training

model.train()
train_losses = []
running_loss = 0.0 

for epoch in range(args.epochs):

    for i, (data, target) in enumerate(train_loader):

        data = data.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        optimiser.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimiser.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(i, loss.item())
            torch.save(model.state_dict(), 'model.pth')
            torch.save(optimiser.state_dict(), 'optimiser.pth')
            torch.save(train_losses, 'train_losses.pth')

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 100,
                            epoch * len(train_loader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(model, data, target),
                            global_step=epoch * len(train_loader) + i)
            running_loss = 0.0

# Testing

model.eval()
test_loss, correct = 0, 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_data)
acc = correct / len(test_data)
print(acc, test_loss)


