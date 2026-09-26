#&& 
import pandas as pd  #essential for data manipulation and analysis
import numpy as np  #essential for numerical computations
import sklearn.preprocessing as preprocessing   #data preprocessing and scaling
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization

'''DATA PREPROCESSING'''
#loading and handling missing values
disease_df = pd.read_csv("framingham.csv") #Read the dataset
disease_df.drop(columns =['education'], inplace = True, axis = 1)  #drop the education column
disease_df.rename( columns = {'male': 'Sex_male'}, inplace = True) #rename the male column to Sex_male

disease_df.dropna( axis = 0, inplace = True) #remove any rows with missing values (NaN) from the Data Frame
#disease_df

print(disease_df.TenYearCHD.value_counts()) 

'''SPLIT DATA INTO TRAINING AND TESTING SETS'''
x = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', ]])
y = np.asarray(disease_df['TenYearCHD'])

x = preprocessing.StandardScaler().fit(x).transform(x)   
#this scales the features in X to have a mean of ) and standard deviation of 1 using StandardScaler. 
#Scaling is important for many machine learning models, especially when the features have different units or magnitudes

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.3, random_state = 4)
print('Train test:', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)

'''Exploratory Data Analysis'''

plt.figure(figsize = (7,5))
sns.countplot( x = 'TenYearCHD', hue = 'TenYearCHD',
              data = disease_df, palette = 'BuGn_r', legend = False)
#create a count plot using SeaBorn which visualize the distribution of the values in the TenYearCHD column showing how many individuals have heart disease (1), and don't (0)
plt.show()
#right click on this file to "Run in interactive window"

#Count number of patients affected by CHD (0 = not affected, 1 = affected)
laste = disease_df['TenYearCHD'].plot()
plt.show(laste)

'''Fitting Logistic Regression Model'''
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression() #create an instance of the Logistic Regression model
logreg.fit(x_train, y_train)  #train the logistic regression model using training data (x_train for descriptive features and y_train for target features)
y_pred = logreg.predict(x_test)  #use train LR to make prediction on test set (x). and predicted values stored in y_pred

'''Evaluating the Logistic Regression Model'''
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =', accuracy_score(y_test, y_pred))

'''Plot confusion matrix'''
from sklearn.metrics import confusion_matrix, classification_report

print('The details for confusion matrix is = ')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred) 
#compute confusion matrix by comparing the actual values (y_test) with the predicted values (y_pred)
#it returns a 2x2 matrix showing true positives, true negatives, false positives and false negatives

conf_matrix = pd.DataFrame(data = cm,
                           columns = ['Predicted:0', 'Predicted:1'],
                           index = ['Actual:0', 'Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap( conf_matrix, annot = True, fmt = 'd', cmap = 'Greens')
plt.show()