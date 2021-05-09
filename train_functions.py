# Import modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os 
import os 
import json
import pandas as pd
import numpy as np

'''
Pre-trained model architecture function
'''
def model_arch(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True) 
        
    for parameters in model.parameters():
        parameters.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units,102),
        nn.LogSoftmax(dim=1)
        )

    model.classifier = classifier
        
    return model


'''
Validation Function
'''
def validation(model, criterion, validloader, device):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        
        # change images and labels to cuda if available

        images, labels = images.to(device), labels.to(device) 

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


'''
Train function
'''
def train_network(epochs, trainloader, validloader, model, optimizer, criterion, device):
    print_every = 50
    steps = 0
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in iter(trainloader):
            steps += 1
            
            # change images and labels to cuda if available
            images, labels = images.to(device), labels.to(device)  
            
            # reset gradients to zeros
            optimizer.zero_grad()

            # feed-Forward and Backpropagation
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # update the weights
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                # evaluation mode
                model.eval()

                # Turning off the gradients for validation
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, criterion, validloader, device)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # turn back train mode
                model.train()

    return model


'''
Load data function
'''
def load_data(train_dir, valid_dir, test_dir ):
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(45),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

                        'valid_test': transforms.Compose([transforms.Resize(254),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])
                      }

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_test']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['valid_test'])
                     }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
              }
    
    return image_datasets, dataloaders


'''
Save the checkpoint function
'''
def save_checkpoint(save_dir, checkpoint_file, model, criterion, optimizer, epochs, cat_to_name, image_datasets):

    # Mapping classes to Indices
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'cat_to_name': cat_to_name,
                  'class_to_idx': model.class_to_idx,
                  'model': model,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'criterion': criterion,
                  'optimizer': optimizer,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs
                 }
    # Check if saving directory doesn't exist then create it
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Save the checkpoint    
    torch.save(checkpoint, save_dir+'/'+checkpoint_file)