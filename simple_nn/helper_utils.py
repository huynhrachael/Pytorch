import torch
import matplotlib.pyplot as plt

def plot_results(model, distances, times):
    """
    Plots actual datra points and the model's predicted line for a given dataset.
    
    Args:
        model: The trained machine learning model to use for predictions.
        distances: input data points (features) for the model.
        times: the target data points (labels) for the plot.
    """ 
    #Set the model to evaluation mode
    model.eval()
    
    #disable gradient calculation for efficient inference
    with torch.no_grad(): #clear values from previous iterations
        #make predictions using the trained model
        predicted_times = model(distances)
        
    #create a new figure for the plot
    plt.figure(figsize=(8,6))
    
    #plot the actual data points
    plt.plot(distances.numpy(), times.numpy(), color = "orange", marker = "o", linestyle = "None", label = "Actual delivery times")
    
    #plot the predicted line from the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color = "blue", marker = "None",label = "Predicted line")
    
    #set the title of the plot
    plt.title("actual vs predicted delivery times")
    #set the x-axis label
    plt.xlabel("Distance (miles)")
    #set the y-axis label
    plt.ylabel("Time (minutes)")
    #display the legend on the plot
    plt.legend()        
    #add a grid to the plot
    plt.grid(True)
    #show the plot
    plt.show()
    
def plot_linear_comparison(model, new_distances, new_times):
    """
    Compares and plot the prediction of a model against new, non-linear data
    
    Args: 
        model: the trained model to be evaluated
        new_distances: new input  data for generating predictions
        new_times: the actual target values for comparison
    """
    #set the model to evaluation mode
    model.eval()
    
    #disable gradient computation for inference
    with torch.no_grad():
        #generate predictions using the model
        predictions = model(new_distances)
        
    #create a new figure for the plot
    plt.figure(figsize=(8,6))
    
    #plot the actual data points
    plt.plot(new_distances.numpy(), new_times.numpy(), color = "orange", marker = "o", linestyle = "None", label = "Actual data (bikes & cars)")
    
    #plot the prediction from the model
    plt.plot(new_distances.numpy(), predictions.numpy(), color = "green", marker = "None", label = "Linear model prediction")
    
    #set the title of the plot
    plt.title("linear Model vs. Non-linear Reality")
    #set the label for the x-axis
    plt.xlabel("Distance (miles)")
    #set the label for the y-axis
    plt.ylabel("Time (minutes)")
    #add a legend to the plot
    plt.legend()    
    #add a grid to the plot for better readability
    plt.grid(True)
    #display the plot
    plt.show()
    
    
