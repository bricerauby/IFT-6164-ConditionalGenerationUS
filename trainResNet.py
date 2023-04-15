from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tqdm
import torchvision
import torchvision.transforms as transforms
from trainer.functionnalTrainingClassifier import train, test, adjust_learning_rate
import os
import glob
from models import *

learning_rate = 0.1
momentum = 0.9
weight_decay=0.0002
batch_size_train= 128
batch_size_eval=400
device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment = Experiment(project_name='cgenulm',
                            workspace='bricerauby', auto_log_co2=False)
experiment.set_name(os.environ.get('SLURM_JOB_ID') + '_' + experiment.get_name())
code_list = glob.glob("**/*.py", recursive=True)

for code in code_list:
    experiment.log_code(file_name=code)


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=6)

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch, learning_rate)
    train(train_loader, net, epoch, experiment, optimizer,criterion, device)
    test(test_loader, net, epoch, experiment,criterion, device)