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
`cd ./src`
`python main.py`

the best model will be stored in models folder under the name resnet32_Mhana_cifar_bestmodel.pt

- to test the model on a random group of images in the test set
`cd ./src`
`python predict_random_samples.py`

- to inference only for one image:
`cd ./src`
`python inference.py --sample_path './../sample.jpeg'`
 
 ## Trained models
You can find the fully trained models which I have trained on CIFAR10 in this link
https://drive.google.com/drive/folders/1dzcB8KHD1y6DUzrTyzBRe-vKO6sVFtqG?usp=sharing


 ## deployment
The best training model has been deployed as back end service on Heroku:
https://deeplearning-ai-project.herokuapp.com/

example how to use it with GET request:
https://deeplearning-ai-project.herokuapp.com/predict?imageURL=https://upload.wikimedia.org/wikipedia/commons/c/c9/Cat.jpeg

and a frontend app has been develop it to use the model in this link:
https://ai-project-frontend-h1j1co68j-anasm87.vercel.app/
