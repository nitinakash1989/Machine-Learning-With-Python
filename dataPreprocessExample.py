#Importing Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Data.csv")

#Creating matrix of feature

#Independent variable
X = dataset.iloc[:,:-1].values

#Dependendent variable
y = dataset.iloc[:,3].values

#Handling missing values in dataset
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy= 'mean', axis= 0)

#To fit imputer in matrix of feature X
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])