# This is the training script to train the model
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

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
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model
checkpoint = torch.load(os.path.join('outputs', 'model.pth'))
# Load the model.
model = build_model(
                    num_classes=len(CLASS_NAMES)
                    ).to(DEVICE)
    
model.load_state_dict(checkpoint['model_state_dict']) # loda the saved model
    
model.eval()
    

def plotshap(model,test_images,background):
    """
    explainer = shap.DeepExplainer(model, image)
    shap_values = explainer.shap_values(image)
    #explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(X_test)
    print("Variable Importance Plot - Global Interpretation")
    figure = plt.figure()
    shap.summary_plot(shap_values, image)
    shap.summary_plot(shap_values[0], image)
    plt.savefig('outputs/shape.png')
    
    """
    print("----------------------------------Welcome to Interpretability ----------------------------\n")
    e = shap.DeepExplainer(model, background)
    print("Shape:",background.shape)
    print(test_images.shape)
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy)


def classifier_fn (numpy_image):
    torch_image = torch.from_numpy(numpy_image)
    with torch.no_grad():
        outputs = model (torch_image)
        _ , predicted_class = torch.max(outputs, 1)
        probs = F.softmax(predicted_class, dim=1)
    return probs.detach().cpu().numpy()
    

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    # Define data loaders for validation set
    validloader = torch.utils.data.DataLoader(
                          dataset_valid,
                          batch_size=batch_size)
   
    """
    batch = next(iter(validloader))
    images, _ = batch
    
    #background = images[:1].to(DEVICE)
    #test_images = images[1:10].to(DEVICE)
    #plotshap(model,test_images,background) # Call to plot intepretability SHAP for the 10 first images
    """  
   
    #define the image and proceed with classification and interpretability
    image_path = "input/Validation/Bleeding/img- (10).png"
    input_size = 224
    test_transform = transforms.Compose([transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.094, 0.0815, 0.063], [0.1303, 0.11, 0.0913])])
    
    image = Image.open(image_path) # read the image from its location
    input_image = test_transform(image).unsqueeze(0)
    
    print ("Image shape:",input_image.shape)
    input_image_numpy = input_image[0].permute(1,2,0).numpy()
    input_image_numpy= input_image_numpy.astype("double")

    explainer = lime_image.LimeImageExplainer()  
    explanation = explainer.explain_instance(input_image_numpy, classifier_fn, top_labels=5, num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=False)
    
    # Visualize explanation
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.axis('off')
    plt.show()
    the_name=Path(image_path).stem # get the name of the file
    the_name="Line"+the_name
    plt.savefig('outputs/'+the_name+'.png')
    
    