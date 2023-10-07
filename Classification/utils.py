import torch
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.style.use('ggplot')

def save_model(model, optimizer, criterion):
    """
    Function to save the trained model to disk in the outputs directory.
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join('outputs', 'model.pth'))

def save_plots(train_acc, valid_acc, train_loss, valid_loss,fold):
    """
    Function to save the loss and accuracy plots to disk in the outputs directory.
    """
    thetitleA='Accuracy-'+str(fold)
    thetitleL='Loss-'+str(fold)
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel(thetitleA)
    plt.legend()
    plt.savefig(os.path.join('outputs', thetitleA+'.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel(thetitleL)
    plt.legend()
    plt.savefig(os.path.join('outputs', thetitleL+'.png'))
    
   
    