import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os as os

# define resnet32 module
class Resnet32Mhana(nn.Module): 
  
  def __init__(self,num_classes=2):
    super(Resnet32Mhana,self).__init__()


    self.relu = nn.ReLU(inplace=True)
    # layers 
    # layer 0
    self.layer0 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride = 2, padding=1)
      )

    # layer 1
    self.layer1_block1 = nn.Sequential(
          nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1, dilation=1,bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1, dilation=1,bias=False),
          nn.BatchNorm2d(64)
        )
    self.layer1_block2 = nn.Sequential(
          nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(64)
        )
    self.layer1_block3 = nn.Sequential(
          nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(64)
        )
    
    
    # layer 2
    self.layer2_downsample = nn.Sequential(
          nn.Conv2d(64,128,kernel_size=1,stride=2,dilation=1,bias=False),
          nn.BatchNorm2d(128)
        )
    
    self.layer2_block1 = nn.Sequential(
          nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128)
        )
    self.layer2_block2 = nn.Sequential(
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128)
        )
    self.layer2_block3 = nn.Sequential(
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128)
        )
    self.layer2_block4 = nn.Sequential(
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(128)
        )

    # layer 3
    self.layer3_downsample = nn.Sequential(
          nn.Conv2d(128,256,kernel_size=1,stride=2,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )
    
    self.layer3_block1 = nn.Sequential(
          nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )
    self.layer3_block2 = nn.Sequential(
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )
    self.layer3_block3 = nn.Sequential(
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )
    self.layer3_block4 = nn.Sequential(
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )
    self.layer3_block5 = nn.Sequential(
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )
    self.layer3_block6 = nn.Sequential(
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(256)
        )

    # layer 4
    self.layer4_downsample = nn.Sequential(
          nn.Conv2d(256,512,kernel_size=1,stride=2,dilation=1,bias=False),
          nn.BatchNorm2d(512)
        )
    
    self.layer4_block1 = nn.Sequential(
          nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(512)
        )
    self.layer4_block2 = nn.Sequential(
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(512)
        )
    self.layer4_block3 = nn.Sequential(
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
          nn.BatchNorm2d(512)
        )
    
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512,num_classes)
    self._init_params()



  def _init_params(self):
      # we should initialize the convolutions according to the activation function
      # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)



   

    
    
  def forward(self, x):
    #layer0
    x=self.layer0(x)

    ##layer1:
    #block1:
    input_layer1_block1 = x
    x = self.layer1_block1(x)
    x += input_layer1_block1
    x = self.relu(x)

    #block2:
    input_layer1_block2 = x
    x = self.layer1_block2(x)
    x += input_layer1_block2
    x = self.relu(x)

    #block3:
    input_layer1_block3 = x
    x = self.layer1_block3(x)
    x += input_layer1_block3
    x = self.relu(x)

    ##layer2
    #block1:
    # we apply here downsampling on the input to be able to added to the output
    input_layer2_block1 = self.layer2_downsample(x)
    x=self.layer2_block1(x)
    x += input_layer2_block1
    x = self.relu(x)

    #block2:
    input_layer2_block2 = x
    x=self.layer2_block2(x)
    x += input_layer2_block2
    x = self.relu(x)

    #block3:
    input_layer2_block3 = x
    x=self.layer2_block3(x)
    x += input_layer2_block3
    x = self.relu(x)

    #block4:
    input_layer2_block4 = x
    x=self.layer2_block4(x)
    x += input_layer2_block4
    x = self.relu(x)

    ##layer3
    #block1
    # we apply here downsampling on the input to be able to added to the output
    input_layer3_block1 = self.layer3_downsample(x)
    x=self.layer3_block1(x)
    x += input_layer3_block1
    x = self.relu(x)

    #block2:
    input_layer3_block2 = x
    x=self.layer3_block2(x)
    x += input_layer3_block2
    x = self.relu(x)

    #block3:
    input_layer3_block3 = x
    x=self.layer3_block3(x)
    x += input_layer3_block3
    x = self.relu(x)

    #block4:
    input_layer3_block4 = x
    x=self.layer3_block4(x)
    x += input_layer3_block4
    x = self.relu(x)

    #block5:
    input_layer3_block5 = x
    x=self.layer3_block5(x)
    x += input_layer3_block5
    x = self.relu(x)

    #block6:
    input_layer3_block6 = x
    x=self.layer3_block6(x)
    x += input_layer3_block6
    x = self.relu(x)

    ##layer4
    #block1:
    # we apply here downsampling on the input to be able to added to the output
    input_layer4_block1 = self.layer4_downsample(x)
    x=self.layer4_block1(x)
    x += input_layer4_block1
    x = self.relu(x)
    
    #block2:
    input_layer4_block2 = x
    x=self.layer4_block2(x)
    x += input_layer4_block2
    x = self.relu(x)

    #block3:
    input_layer4_block3 = x
    x=self.layer4_block3(x)
    x += input_layer4_block3
    x = self.relu(x)

    # fully connected layer 
    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.fc(x)





    return x
