
import torch #pytorch's core functionality
import torch.nn as nn #components for building neural networks
import torch.optim as optim #tools to train those models

import helper_utils 

#this line ensures that your results are reproducible an d consistent every time
torch.manual_seed(42)

#step 1 & 2: DATA INGESTION AND PREPARATION
#distance in miles and for recent bike delivery 
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

#step 3: MODEL BUILDING 
#single neuron with one input implement a linear equation: Time = Weight * Distance + Bias
#CREATE A LINEAR MODEL
model = nn.Sequential(nn.Linear(1,1)) #one input and one output

#step 4: MODEL TRAINING
#define loss function and optimizer
loss_function = nn.MSELoss() #mean squared error loss function
optimizer = optim.SGD(model.parameters(), lr=0.01) #stochastic gradient descent, adjust weight and bias based on errors

#TRAININ LOOP (500 EPOCHS)
for epoch in range(500):
    #reset the optimizer's gradients
    optimizer.zero_grad()
    
    #make predictions (forward pass)
    outputs = model(distances)
    
    #calculate the loss
    loss = loss_function(outputs, times)
    
    #calculate adjustments (backward pass)
    loss.backward()
    
    #update the model's parameters
    optimizer.step()
    
    #print loss every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.items()}")
        
#plot_results will show original data points (actual deliveries), the line your model learned (its predictions), and how well they match
helper_utils.plot_predictions(model, distances, times)

distance_to_predict = 7.0 

#use the torch.no_grad() context manager for efficient predictions
with torch.no_grad():
    #convert python variable into a 2D pytorch tensor that the mode expects
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    
    #pass the new data to the trained model to get a prediction
    predicted_time = model(new_distance)
    
    #use .item() to extract th scalar value from the tensor for printing
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes")
    
    #use the scalar value in a conditional statement make the final decision
    if predicted_time.item() > 30:
        print("\nDecision: Do not take the job. You'll likely to be late.")
    else:
        print("\nDecision: Take the job. You can make it! ")
        
#Inspecting the model's learning
#accesss the first (and only) layer in the sequential model
layer = model[0]

#get weights and bias
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weights: {weights}, Bias: {bias}")

#TEST MODEL ON MORE COMPLEX DATA 
#combined dataset: bikes for short distances, and cars for longer distances
new_distances = torch.tensor([[1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5], 
                              [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
                              [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
                              [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]], dtype=torch.float32)     

#corresponding delivery times in minutes
new_times = torch.tensor([[6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
                          [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
                          [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
                          [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]], dtype=torch.float32)

#use already-trained linear model to make predictions
with torch.no_grad():
    predictions = model(new_distances)

#calculate the new loss
new_loss = loss_function(predictions, new_times)
print(f"Loss on new, combined data: {new_loss.item():.2f}")

helper_utils.plot_predictions(model, new_distances, new_times)