# Imports functions created for this program
from predict_functions import *
from user_input_args import user_input_args_2

# Import python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json


# Main program function defined below
def main():
    # Get user inputs
    in_arg = user_input_args_2()
    
    # Set inputs into variables
    
    # path to image wich will be predicted
    input_ = in_arg.input
    # path to checkpoint used for predciting process
    checkpoint = in_arg.checkpoint
    # top K most likely classes
    top_k = in_arg.top_k
    # mapping of categories to real names file 
    category_names = in_arg.category_names
    
    # GPU for training
    gpu = in_arg.gpu

    # Load label mapping
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
     
    # process image and turn it to pytorch tensor
    pytorch_tensor = process_image(input_)   
    
    # Load checkpoint
    model, criterion, optimizer = load_checkpoint(checkpoint, gpu=gpu)
    
    
    # Enable cuda if available
    if gpu: 
        device = "cuda" 
    else:
        device = "cpu"
    
    # Change model and tensor to cuda if available
    pytorch_tensor = pytorch_tensor.to(device)
    model = model.to(device)
    
    # Predict flower name along with probability of the name
    probs, classes = predict(pytorch_tensor, model, topk=top_k)
    
    # Print outputs
    print('')
    print('{:30} {}\n'.format('Flower name:' , 'Class Probability:'))
    if round(probs[0], 2) == 1:
        print('{:30} {}%'.format(cat_to_name[classes[0]], round((probs[0]*100), 2)))
    else:     
        for _class, prob in zip(classes,probs):
            print('{:30} {}%'.format(cat_to_name[_class], round((prob*100), 2)))


# Call to main function to run the program
if __name__ == "__main__":
    main()     