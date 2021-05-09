# Imports functions created for this program
from train_functions import *
from user_input_args import user_input_args

# Import python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os 
import json
import pandas as pd
import numpy as np

# Main program function defined below
def main():
    # Get user inputs
    in_arg = user_input_args()
    
    # Set inputs into variables
    
    # data directories
    train_dir = in_arg.data_dir
    valid_dir = 'flowers/valid'
    test_dir = 'flowers/test'
    
    # Save checkpoints 
    save_dir = in_arg.save_dir
    checkpoint_file = in_arg.checkpoint
    
    # Model architecture 
    arch = in_arg.arch
    
    # Hyperparameters inputs for Model
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs 
    
    # GPU for training
    gpu = in_arg.gpu
    
    # Load label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Load Data for training and validation
    image_datasets, dataloaders = load_data(train_dir, valid_dir, test_dir )
    trainloader, validloader, testloader= dataloaders['train'], dataloaders['valid'], dataloaders['test']

    # Build the model architecture
    model = model_arch(arch, hidden_units)
    
    # Criterion
    criterion = nn.NLLLoss()

    # Define the optimizer the classifier parameters
    optimizer= optim.Adam(model.classifier.parameters(),lr=learning_rate)
    
    
    # Enable cuda if available
    if gpu: 
        device = "cuda" 
    else:
        device = "cpu"

    # Change model to cuda if available
    model = model.to(device) 
    
    # Train the network
    model = train_network(epochs, trainloader, validloader, model, optimizer, criterion, device)
    
    # Save checkpoint for the trained model
    save_checkpoint(save_dir, checkpoint_file, model, criterion, optimizer, epochs, cat_to_name, image_datasets)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()

