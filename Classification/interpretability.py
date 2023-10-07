# This is the training script to train the model
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from numpy import asarray
from model import build_model
from datasets import get_datasets
from utils import save_model, save_plots
import shap
batch_size = 20
import numpy as np
import os
from pathlib import Path

import torch.nn.functional as F
from lime import lime_image
from torchvision import transforms
from PIL import Image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from swin_transformer_pytorch.swin_transformer import SwinTransformer

#explainer = lime_image.LimeImageExplainer()

CLASS_NAMES = ['Bleeding', 'Non-Bleeding']    
#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE='cpu'

checkpoint = torch.load(os.path.join('outputs', 'model.pth'))
    
# Load the model.
model = build_model(
                    num_classes=len(CLASS_NAMES)
                    ).to(DEVICE)
    
model.load_state_dict(checkpoint['model_state_dict']) # load the saved model
 
def model_inference(numpy_array):
    print("-------------------------------------------------------------") 
    torch_image = torch.from_numpy(numpy_array)
    torch_image = torch_image.permute(0,3,1,2)
    model.to(torch.double)
    with torch.no_grad():
        output = model(torch_image.to(DEVICE))
        
        # Softmax probabilities.
        predictions = torch.softmax(output, dim=1).cpu().numpy()
        # Predicted class number.
        output_class = np.argmax(predictions)
                
        print("predictions:",predictions)
        #print("output_class:",output_class)
        print("Classes:",CLASS_NAMES[output_class])
        
        #_, predicted_class = torch.max(output, 1)
    #probs = F.softmax(predicted_class.float(), dim=0)
    #print("predicted_class:",predicted_class)
    #return probs.detach().cpu().numpy().tolist()
    #return CLASS_NAMES[output_class] 
    return predictions
        
    

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    # Define data loaders for validation set
    validloader = torch.utils.data.DataLoader(
                          dataset_valid,
                          batch_size=batch_size)
   
    
    #define the image and proceed with classification and interpretability
    image_path = "input/Test/TestDataset2/A0216.png"
    input_size = 224
    test_transform = transforms.Compose([transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.094, 0.0815, 0.063], [0.1303, 0.11, 0.0913])])
    
    image = Image.open(image_path) # read the image from its location
    input_image = test_transform(image).unsqueeze(0)
    
    print ("Image shape:",input_image.shape)
    input_image_numpy = input_image[0].permute(1,2,0).numpy()
    print ("Image shape1:",input_image_numpy.shape)
    input_image_numpy= input_image_numpy.astype("double")
    print ("Image shape2:",input_image_numpy.shape)
    
    explainer = lime_image.LimeImageExplainer()  
    explanation = explainer.explain_instance(input_image_numpy, model_inference, top_labels=2, num_samples=1)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=False)
   
    # Visualize explanation
    the_name=Path(image_path).stem # get the name of the file
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.axis('off')
    plt.title(label=the_name, 
          fontsize=20, 
          color="green") 
    plt.show()
    the_name="LimeTestDataset2"+the_name
    plt.savefig('outputs/'+the_name+'.png')
 

    