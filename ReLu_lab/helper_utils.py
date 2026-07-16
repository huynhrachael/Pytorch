import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def plot_data (distances, times, normalize = False):
    """
    Create a scatter plot of  the data points

    Args:
        distances: The input data points for the x-axis
        times: The corresponding time values for the y-axis
        normalize: A boolean flag indicating whether the data is normalised
    """
    
    #create a new figure with a specified size
    plt.figure(figsize=(8,6))
    
    #plot the data points as a scatter plot
    plt.plot(distances.numpy(), times.numpy(), color = 'orange', marker = 'o', linestyle = 'None', label = 'Actual delivery times')
    
    #check if the data is normalised to set appropriate labels and titles
    if normalize:
        #set plot tile for normalised data
        plt.title('Normalised delivery data (bikes and cars)')
        
        #set x-axis label for normalized data
        plt.xlabel('Normalized distance')
        
        #set the y-axis label for normalized data
        plt.ylabel('Normalized time')   
        
        #display the legend
        plt.legend()
        
        #add a grid to the plot
        plt.grid(True)
        
        #show the plot
        plt.show()
    #Handle the case for un-normalized data
    else:
        #set the plot title for un-normalized data
        plt.title('Delivery data (bikes and cars)')
        
        #set the x-axis label for un-normalized data
        plt.xlabel('Distance (miles)')
        
        #set the y-axis label for un-normalized data
        plt.ylabel('Time (minutes)')
        
        #display the legend
        plt.legend()
        
        #add a grid to the plot
        plt.grid(True)
        
        #show the plot
        plt.show()

def plot_final_fit(model, distances, times, distance_norm, times_std, times_mean):
    """
    PLots the prediction of a trained model aganist the original data,
    after de-normalizing the predictions
    
    Args:
        model: the trained model used for predictions
        distances: the original, un-normalized input data
        times: the original, un-normalized target data
        distance_norm: the normalized input data for the model
        times_std: the standard deviation used for de-normalization
        times_mean: the mean value used for de-normalization
    """
    #set the model to evaluation mode
    model.eval()
    
    #disable gradient calculations for predictions
    with torch.no_grad():
        #get predictions from the model using normalized data
        predicted_norm = model(distance_norm)
    
    #de-normalize the predictions to their original scale
    predicted_times = (predicted_norm * times_std) + times_mean
    
    #create a nre figure for the plot
    plt.figure(figsize=(8,6))
    
    #plot the original data points
    plt.plot(distances.numpy(), times.numpy(), color = 'orange', marker = 'o', linestyle = 'None', label = 'Actual data (bikes and cars)')
    
    #plot the denormalized predictions form the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color = 'green', label = 'Non-linear Model Predictions')
    
    #set the title of the plot
    plt.title('Non-Linear Model Fit vs. Actual Data')
    
    #set the x-axis label
    plt.xlabel('Distance (miles)')
    
    #set the y-axis label
    plt.ylabel('Time (minutes)')
    
    #add the legend to the plot
    plt.legend()
    
    #enable the grid
    plt.grid(True)
    
    #display the plot
    plt.show()
    
def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    """
    Plot the training progress of a model on normalized data,
    showing the current fit at each epoch

    Args:
        epoch: The current epoch number
        loss: the loss value at the current epoch
        model: the trained model
        distances_norm: the normalized input data
        times_norm: the normalized target data
    """
    loss = 0
    #clear the previous plot form the output cell
    clear_output(wait=True)
    
    #make predictions using the current state of the model
    predicted_norm = model(distances_norm)
    
    #convert tensors using the current state of the model
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()
    
    #detach predictions from the computation graph and convert to Numpy
    y_pred_plot = predicted_norm.detach().numpy()
    
    #sort the data based on distanc to ensure a smooth line plot
    sorted_indices = x_plot.argsort(axis=0).flatten()
    
    #create a new figure for the plot
    plt.figure(figsize=(8,6))
    
    #plot the original normalized data points
    plt.plot(x_plot, y_plot, color = 'orange', marker = 'o', linestyle = 'None', label = 'Actual normalized data')
    
    #plot the model's prediction as a line
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices], color = 'green', label = 'Model Predictions')
    
    #set the title of the plot, include the current epoch
    plt.title(f'Epoch: {epoch + 1} | Normalized training progress')
    
    #set the x-axis label
    plt.xlabel('Normalized distance')
    
    #set the y-axis label
    plt.ylabel('Normalized time')
    
    #display the legend
    plt.legend()
    
    #add a grid to the plot
    plt.grid(True)
    
    #display the plot
    plt.show()
    
    #pause briefly to allow the plot to rendered
    time.sleep(0.6)
    