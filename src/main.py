
#import libraries

import numpy as np
import matplotlib.pyplot as plt
from module import *
from train import *
from helper import *
import os

#setup tensor board 
from torch.utils.tensorboard import SummaryWriter


def main_resnet32_mhana():
    # log_dir is "runs" where we log valication loss, training loss and accuracy
    writer = SummaryWriter('./../runs')



    # seeding
    set_seed(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # configure device to gpu if it exists
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    #download cifar10 training set and comput the mean and the standard deviation
    train = torchvision.datasets.CIFAR10(root= CIFAR10_DATA_PATH, train=True, download=True)
    DATA_MEANS = (train.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (train.data / 255.0).std(axis=(0,1,2))

    #define test transformations: 1.transform data to tensors 2. normalize using training data mean and std
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(DATA_MEANS, DATA_STD)
                                        ])
    # Networks are too powerful and would overfit. that's why For training, we add some augmentation. 
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(DATA_MEANS, DATA_STD)
                                        ])
    #batch size 
    batch_size = 128

    #training deta set
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_DATA_PATH, train=True,download=True, transform=train_transform)
    #validation daatset , we apply a trick because the transform for validation set is differnt than training
    valset = torchvision.datasets.CIFAR10(root=CIFAR10_DATA_PATH, train=True, transform=test_transform, download=True)
    set_seed(42)
    train_set, _ = torch.utils.data.random_split(trainset, [45000, 5000])
    set_seed(42)
    _, val_set = torch.utils.data.random_split(valset, [45000, 5000])
    #test set 
    testset = torchvision.datasets.CIFAR10(root=CIFAR10_DATA_PATH, train=False,download=True, transform=test_transform)

    #data loaders for every set
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)



    #define our module 
    resnetm32=Resnet32Mhana(num_classes=10)

    #move the module to gpu if avalible
    resnetm32.to(device)

    #define our optimizer
    optimizer = optim.SGD(resnetm32.parameters(), lr=0.1, momentum= 0.9, weight_decay=1e-4)

    # We will reduce the learning rate by 0.1 gamma
    # that's why we define our secheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 25], gamma=0.1)

        
    
     
    train_module(resnetm32, optimizer,scheduler,torch.nn.CrossEntropyLoss(), trainloader, valloader, testloader,writer= writer, epochs=40,
        device=device, path = MODEL_PATH )

    writer.flush()
    writer.close()

if __name__=='__main__':
    main_resnet32_mhana()