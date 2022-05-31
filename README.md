# AI engineering course project

This repo contains the implementation of ResNet32 

## Requirements
python-3.9.0
The most important packages are pytorch, torchvision, numpy and matplotlib.
(Works with Pytorch 1.11.0)

## Dataset
cifar10
https://www.cs.toronto.edu/~kriz/cifar.html

## Run Code
after following the commands in setup(ps for windows, sh for linux)
To run the code:

- training your model resnet32 from scratch
`python main.py`

the best model will be stored in models folder under the name resnet32_Mhana_cifar_bestmodel.pt

- to test the model on a random group of images in the test set
`python predict_random_samples.py`

- to inference only for one image:
`python inference.py --sample_path './../sample.jpeg'`
 
 ## Trained models
You can find the fully trained models on CIFAR10 in this link

