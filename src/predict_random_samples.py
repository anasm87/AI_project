
#import libraries

import numpy as np
import matplotlib.pyplot as plt
from module import *
from train import *
from helper import *
import os
#setup tensor board 
from torch.utils.tensorboard import SummaryWriter

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def predict_samples():


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

    #batch size 
    batch_size = 4


    #test set 
    testset = torchvision.datasets.CIFAR10(root=CIFAR10_DATA_PATH, train=False,download=True, transform=test_transform)

    #data loaders for every set
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
        optimizer, milestones=[20, 30], gamma=0.1)

        
    
    #upload the best model if exist and start from there
    losslogger =[]
    BEST_MODEL_='resnet32_Mhana_cifar_bestmodel.pt'
    path_best_model = os.path.join(MODEL_PATH,BEST_MODEL_)
    model, optimizer,scheduler, start_epoch, losslogger= load_checkpoint(resnetm32, optimizer,scheduler, losslogger, filename=path_best_model)
    
    #move the model and the optimizer to the gpu if it exists 
    model.to(device)

    # get some random training images
    dataiter = iter(testloader)

    #randomly go in the test dataset
    x = random.randint(1,len(testloader))
    for i in range(x):
      dataiter.next()

    images, labels = dataiter.next()
    preds, probs = images_to_probs(model, images)

    fig = plot_classes_preds(model, images, labels)
    fig.savefig('random_samples_predictions.jpg')
    img_grid = torchvision.utils.make_grid(images)

    fig1, ax1 =  plt.subplots()
    ax1.imshow(img_grid.permute(1, 2, 0))
    fig1.savefig('random_samples_image_grid.jpg')



if __name__=='__main__':
    predict_samples()