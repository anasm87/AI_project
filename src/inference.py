
#import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from module import *
from train import *
from helper import *
import os
#setup tensor board 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from PIL import Image


import argparse
import os
import sys

# Training setttings
parser = argparse.ArgumentParser(description='sample prediction task')
parser.add_argument('--sample_path', required=True, help='path to the sample')

def predicte(filepath):

 
    # configure device to gpu if it exists
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

 
    #mean and std are computed from the training dataset
    DATA_MEANS = [0.49139968, 0.48215841, 0.44653091]
    DATA_STD = [0.24703223, 0.24348513, 0.26158784]



    #define test transformations: 1.transform data to tensors 2. normalize using training data mean and std
    test_transform = transforms.Compose([transforms.Resize((32,32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=DATA_MEANS, std=DATA_STD)
                                        ])

                                    

    #define our module 
    resnetm32=Resnet32Mhana(num_classes=10)

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
    model.eval()

    #predict
    img = Image.open(filepath)
    img = test_transform(img)
    img = img.unsqueeze(0)

    img=img.to(device)


    output = model(img)

    
    # predicted class 
    prediction = output.argmax()
    predicted_class = classes[prediction]

    props = F.softmax(output, dim=1).cpu().data.numpy().squeeze()
    prediction_probability = props[props.argmax()]
   
    print('predicted class is {} with probability {}'.format(predicted_class,prediction_probability * 100.0))

    return predicted_class, prediction_probability



if __name__=='__main__':
  global args
  args = parser.parse_args()
  if not args.sample_path:
    args.sample_path = r'./../sample.jpeg'
  filepath = args.sample_path
  prediction,_ = predicte(filepath)
  