import argparse

def user_input_args():
    
    """
    Create command line arguments for train.py
    
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments 
    parser.add_argument('data_dir', action="store", type = str, 
                        help = 'path to the folder of training images')

    parser.add_argument('--save_dir', type = str, default = 'saved_checkpoints',
                        help = 'path to the folder of saved checkpoints')
    
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth',
                        help = 'checkpoint file name you would save')
    
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                        help = 'the model architecture to use (vgg16 or vgg13)') 
    
    # Hyperparameters inputs
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                        help = 'the learning rate for the optimizer') 
    
    parser.add_argument('--hidden_units', type = int, default = 4069, 
                        help = 'the number of units in hidden layers') 
    
    parser.add_argument('--epochs', type = int, default = 15, 
                        help = 'n epochs for training the model')
    
    # Train on GPU mode
    parser.add_argument('--gpu',  action='store_true', 
                        help = 'set the train on gpu')

    

    return parser.parse_args()

def user_input_args_2():
    
    """
    Create command line arguments for predict.py
    
    """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments 
    parser.add_argument('input', action="store", type=str,
                        help = 'path to the image which will be predicted')
    
    parser.add_argument('checkpoint', action="store", type=str,
                        help = 'path to checkpoint used for predcting image')
    
    parser.add_argument('--top_k', type = int, default = 1, 
                        help = 'top K most likely classes')
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'a mapping of categories to real names file')
    
    # Use GPU for inference
    parser.add_argument('--gpu',  action='store_true', 
                        help = 'set the train on gpu')
    
    return parser.parse_args()