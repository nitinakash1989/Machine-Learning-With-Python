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

