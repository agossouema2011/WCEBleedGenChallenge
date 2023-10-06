# Test file to test the model with new data from the test dataset
import glob as glob
from datasets import *
from train import *
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image  
from matplotlib import pyplot as plt
from PIL import Image

import xlsxwriter
from pathlib import Path
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm.auto import tqdm
from model import build_model
from torch.utils.data import DataLoader
from torchvision import datasets



# Constants and other configurations.
TEST_DIR = os.path.join('input', 'Validation')
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device=DEVICE
IMAGE_RESIZE = 224
NUM_WORKERS = 4 # to not occupy the full capacity of your device, it is recommended to set numworkers
CLASS_NAMES = ['Bleeding', 'Non-Bleeding']


test_images=[]
class_names = ['Bleeding', 'Non-Bleeding']

result = []

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return test_transform

def get_datasets(image_size):
    """
    Function to prepare the Datasets.
    Returns the test dataset.
    """
    dataset_test = datasets.ImageFolder(
        TEST_DIR, 
        transform=(get_test_transform(image_size))
    )
    return dataset_test

def get_data_loader(dataset_test):
    """
    Prepares the training and validation data loaders.
    :param dataset_test: The test dataset.

    Returns the training and validation data loaders.
    """
    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return test_loader

def denormalize(
    x, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def save_test_results(
    tensor, 
    target, 
    output_class, 
    counter, 
    test_result_save_dir
):
    """
    This function will save a few test images along with the 
    ground truth label and predicted label annotated on the image.

    :param tensor: The image tensor.
    :param target: The ground truth class number.
    :param output_class: The predicted class number.
    :param counter: The test image number.
    """
    image = denormalize(tensor).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = target.cpu().numpy()
    cv2.putText(
        image, f"GT: {CLASS_NAMES[int(gt)]}", 
        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    if output_class == gt:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(
        image, f"Pred: {CLASS_NAMES[int(output_class)]}", 
        (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, color, 2, cv2.LINE_AA
    )
    cv2.imwrite(
        os.path.join(test_result_save_dir, 'test_image_'+str(counter)+'.png'), 
        image*255.
    )
    
    
    
    

def test(model, testloader, device, test_result_save_dir):
    """
    Function to test the trained model on the test dataset.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.

    Returns:
        predictions_list: List containing all the predicted class numbers.
        ground_truth_list: List containing all the ground truth class numbers.
        acc: The test accuracy.
    """
    model.eval()
    print('Testing model')
    predictions_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)
            # Softmax probabilities.
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            # Predicted class number.
            output_class = np.argmax(predictions)
            # Append the GT and predictions to the respective lists.
            predictions_list.append(output_class)
            ground_truth_list.append(labels.cpu().numpy())
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

            save_test_results(
                image, 
                labels,
                output_class, 
                counter, 
                test_result_save_dir
            )
    # Copy the CUDA tensor to CPU memory
    cpu_tensorLabels = labels.data.cpu()
    # Convert the CPU tensor to a NumPy array
    numpy_arrayLabels = cpu_tensorLabels.numpy()
    thepred=preds.cpu().numpy()
    recall= recall_score(numpy_arrayLabels, preds.cpu().numpy(), zero_division=0)
    f1score= f1_score(numpy_arrayLabels, preds.cpu().numpy(), zero_division=0)
    acc = 100. * (test_running_correct / len(testloader.dataset))
    return predictions_list, ground_truth_list, acc,recall,f1score
    



# Function to make the prediction on the test dataset 

def model_inference(model):
    
    print("-------------------------------------------------------------") 
    # For each image proceed the testing
    for i in range(len(test_images)):
        image_name = test_images[i]
      
        theimage = Image.open(image_name).convert('RGB')
        print(image_name)
        from torchvision import transforms
        #
        # Create a preprocessing pipeline
        #
        preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        #
        # Pass the image for preprocessing and the image preprocessed
        #
        img_preprocessed = preprocess(theimage)
        #
        # Reshape, crop, and normalize the input tensor for feeding into network for evaluation
        #
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
        model.eval()
        #
        # Get the predictions of image as scores related to how the loaded image
        # matches with 1000 ImageNet classes. The variable, out is a vector of 1000 scores
        #
        output = model(batch_img_tensor.to(device))
        
        index = output.max(dim=1).indices.item()
        print("Classes:",class_names[index])
        the_name=Path(image_name).stem # get the name of the file
        result.append([the_name, class_names[index]])
        print("-------------------------------------------------------------")  
        
        
        
        
# The main function starts from here

if __name__ == '__main__':
    # Set the directory for the data
    # directory where all the images are present
    DIR_TEST = 'input/Test'
    
    workbook = xlsxwriter.Workbook('Results.xlsx') # Create an excel Result file To save the results of the classification
    worksheet = workbook.add_worksheet()
    # Use the worksheet object to write
    # data via the write() method.
    worksheet.write('A1', 'Image ID')
    worksheet.write('B1', 'Predicted class label')

    test_images = glob.glob(f"{DIR_TEST}/TestDataset2/*.png") # for the test image directory selection
    
    #-------------------- This part make the prediction on the validation set and store in a folder called  "test_results" -------------------
    
    test_result_save_dir = os.path.join('outputs', 'test_results') # Create a folder "test_results" to store the prediction images on validation set
    os.makedirs(test_result_save_dir, exist_ok=True)

    dataset_test = get_datasets(IMAGE_RESIZE)
    test_loader = get_data_loader(dataset_test)

    checkpoint = torch.load(os.path.join('outputs', 'model.pth'))
    
    # Load the model.
    model = build_model(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict']) # loda the saved model
    
    predictions_list, ground_truth_list, acc,recall,f1score = test(
        model, 
        test_loader, 
        DEVICE,
        test_result_save_dir
    )
    print(f"Test accuracy on Validation Dataset: {acc:.3f}%, Recall: {recall:.3f}, f1score: {f1score:.3f}")
    
    
    #-------------------------------------------------This part make the prediction test on the TestDataset ------------------------------------------------------------------------
    # Display for each image its class
    
    print("\n\n--------------------------------------CLASSIFICATION RESULTS ON TESTDATASET and save the result in an excel file -----------------------\n")
    print(f"Test instances: {len(test_images)}")
    row = 1
    col = 0
    model_inference(model)
    
    # Build the excel file wil the results-- Iterate over the data and write it out row by row.
    for name, label in (result):
        worksheet.write(row, col, name)
        worksheet.write(row, col + 1, label)
        row += 1
    
    workbook.close()
    






