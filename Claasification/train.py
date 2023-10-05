# This is the training script to train the model
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report, confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import numpy as np
from model import build_model
from datasets import get_datasets
from utils import save_model, save_plots
import shap


BATCH_SIZE = 5
k_folds=3
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser for the training: number of epoch, and learning rate.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=25,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('\n Training')
    train_running_loss = 0.0
    train_running_correct = 0
    totalsize=0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        train_running_loss += loss.item()
        totalsize += labels.size(0)
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        #loss.backward()
        # Update the weights.
        #optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = train_running_correct /totalsize
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, class_names,fold,epoch):
    model.eval() # This sets the model on validation mode
    print('\n Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    totalsize=0
    counter = 0
    recall=0.0
    f1score=0.0
    numpy_arrayLabels=[]
    thepred=[]
    CM=0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            totalsize += labels.size(0)
            valid_running_correct += (preds == labels).sum().item()
            CM+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])
            """
            # call for the function to display the interpretatbility plot
            if fold==(k_folds-1) and epoch==(epochs-1):
                plotshap(model,image)
            """   
            
        tn=CM[0][0]
        tp=CM[1][1]
        fp=CM[0][1]
        fn=CM[1][0]
        acc=np.sum(np.diag(CM)/np.sum(CM))
        sensitivity=tp/(tp+fn)
        recall=sensitivity
        precision=tp/(tp+fp)
        epoch_loss = valid_running_loss / counter
        epoch_acc = valid_running_correct / totalsize
        print("\nEpoch Loss:",epoch_loss," Accuracy:",epoch_acc)
        print('\n Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('\nConfusion Matirx : ')
        print(CM)
        print('\n- Recall : ',sensitivity)
        print('\n- Specificity : ',(tn/(tn+fp)))
        print('\n- Precision: ',(tp/(tp+fp)))
        print('\n- NPV: ',(tn/(tn+fn)))
        f1score=(2*sensitivity*precision)/(sensitivity+precision)
        print('\n- F1-Score: ',f1score)
        print()
    return epoch_loss, epoch_acc,recall,f1score
    

# Define function to display interpretability  plot SHAP 
def plotshap(model,X_test):
    explainer = shap.Explainer(model.eval(), X_test)
    shap_values = explainer(X_test)
    #explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(X_test)
    print("Variable Importance Plot - Global Interpretation")
    figure = plt.figure()
    shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values[0], X_test)
    plt.savefig('outputs/shape.png')

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"Number of training images: {len(dataset_train)}")
    print(f"Number of validation images: {len(dataset_valid)}")
    print(f"Classes: {dataset_classes}")

    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu') # choose to use GPU if available else use CPU
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    
    """
    # Load the model.
    model = build_model(
        pretrained=True,
        fine_tune=True, 
        num_classes=len(dataset_classes)
    ).to(device)
    """
    
     # Load the model.
    model = build_model(num_classes=len(dataset_classes)).to(device)
    
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

   
    
    # Load the training and validation data loaders.
    #train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    
    dataset = ConcatDataset([dataset_train, dataset_valid])
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Start print
    print('--------------------------------')
    # For fold results
    resultsTA = {}
    resultsVA = {}
    recallsFolds= {}
    f1ScoreFolds= {}
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        recalls, f1scores = [], []
        # Print
        print(f'\nFOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=BATCH_SIZE, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=BATCH_SIZE, sampler=test_subsampler)
    
    
    
        # Start the training.
        Tacc=0
        Vacc=0
        Rekall=0
        F1=0
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_acc = train(model, trainloader, 
                                                    optimizer, criterion)
            
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            
            print(f"\nTraining loss: {train_epoch_loss:.3f}, Training Accuracy: {train_epoch_acc:.3f}")
            
            print('-'*50)
            time.sleep(2)
            
            # Call for validation    
            valid_epoch_loss, valid_epoch_acc,Recall,F1score = validate(model, testloader,  
                                                            criterion, dataset_classes, fold,epoch)
            
            valid_loss.append(valid_epoch_loss)
            valid_acc.append(valid_epoch_acc)
        
            print(f"\n Validation Accuracy: {valid_epoch_acc:.3f}")
            
            Tacc+=train_epoch_acc
            Vacc+=valid_epoch_acc
            Rekall+=Recall
            F1+=F1score
            
        # Save the loss and accuracy plots.
        save_plots(train_acc, valid_acc, train_loss, valid_loss,fold)
        # Save the trained model weights.
        save_model(model, optimizer, criterion)  
        
        resultsTA[fold] = Tacc/epochs
        resultsVA[fold] = Vacc/epochs
        recallsFolds[fold]=Rekall/epochs
        f1ScoreFolds[fold]=F1/epochs
        
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------TRAINING ACCURACY------------------')
    sum = 0.0
    for key, value in resultsTA.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average Training Accuracy: {sum*100/len(resultsTA.items())} %')
    print('--------------VALIDATION ACCURACY------------------')
    sum = 0.0
    for key, value in resultsVA.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average Validation Accuracy: {sum*100/len(resultsVA.items())} %')
    print('\n\n------------RECALL--------------------')
    sum = 0.0
    for key, value in recallsFolds.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average Recall: {sum*100/len(recallsFolds.items())} %')
    print('\n\n------------F1 SCORE--------------------')
    sum = 0.0
    for key, value in f1ScoreFolds.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average F1-Score: {sum*100/len(f1ScoreFolds.items())} %')
        
   
   
    print('TRAINING COMPLETE')
    
