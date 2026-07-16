import torch
import numpy as np
import pandas as pd

#TENSOR CREATION
#from python lists
x = torch.tensor([1,2,3])
print("from python lists: ", x)
print("tensor data type: ", x.dtype)

#torch.from_numpy() MEANING convert a numpy array to a pytorch tensor
#from numpy array
numpy_array = np.array([[1,2,3],[4,5,6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)
 
print("Tensor from numpy: \n\n", torch_tensor_from_numpy)

#from Pandas DataFrame 
#Read the data from the CSV file into a Pandas DataFrame
df = pd.read_csv('data.csv')

#Extract the data as a numpy array from the DataFrame
all_values = df.values

#Convert the DataFrame's values to a Pytorch tensor 
tensor_from_df = torch.tensor(all_values)

print("Original DataFrame: \n\n", df)
print("\nTensor from DataFrame: \n\n", tensor_from_df)
print("\nTensor data type: ", tensor_from_df.dtype)
