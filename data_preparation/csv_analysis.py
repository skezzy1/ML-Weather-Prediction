import numpy as np
import pandas as pd 

dataset = pd.read_csv("weather_data.csv")

X = dataset.iloc[: , :].values
y = dataset.iloc[: , 0].values

#print(X)
unique_data_meaning = set(y)
print(unique_data_meaning)
print(X)
