# Import modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

'''
Load checkpoint function
'''
def load_checkpoint(checkpoint, gpu=True):
    
    if gpu and torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
       
    checkpoint = torch.load(checkpoint, map_location=map_location)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.cat_to_name = checkpoint['cat_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Frozen parameters for pre-trained feature network
    for parameters in model.parameters():
        parameters.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.epochs = checkpoint['epochs']
    
    return model, criterion, optimizer


'''
Process image for use in Pytorch function
'''
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im.resize((256,256))
    
    width, height = im.size   # Get dimensions
    new_width, new_height = 224, 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    
    # Convert PIL image to Numpy
    np_image = np.array(im)
    
    # Normalize the images
    np_image = np_image / np_image.max()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalization = (np_image - mean) / std
    
    pytorch_tensor = torch.Tensor(normalization.transpose((2,0,1)))
    
    return pytorch_tensor


'''
Image predection function
'''
def predict(pytorch_tensor, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
 
    # evaluation mode
    model.eval()
    
    with torch.no_grad():
        output = model.forward(pytorch_tensor.unsqueeze_(0))
        ps = torch.exp(output)
        
        # top  K  largest values 
        top_ps, topk = ps.topk(topk)
        
        # reverse class_to_idx to idx_to_class
        idx_to_class = {value : key for (key, value) in model.class_to_idx.items()}
        
        # the highest K probabilities and classes
        probs = top_ps[0].cpu().numpy()
        classes = [idx_to_class[_class] for _class in  topk[0].cpu().numpy()]

    return probs, classes
