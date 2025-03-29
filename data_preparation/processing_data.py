import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("data_preparation/weather_data.csv")

dataset["Date_Time"] = pd.to_datetime(dataset["Date_Time"])
dataset["Year"] = dataset["Date_Time"].dt.year
dataset["Month"] = dataset["Date_Time"].dt.month
dataset["Day"] = dataset["Date_Time"].dt.day

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ["Location"])], remainder='passthrough')
X = np.array(ct.fit_transform(dataset.drop(columns=["Date_Time", "Precipitation_mm"])))

y = dataset["Precipitation_mm"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

standartscaler = StandardScaler()
pd.DataFrame(X_train).to_csv("X_train_processed.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test_processed.csv", index=False)

pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("Data preparation completed. Processed files saved.")

print("Data preparation completed. Processed files saved as 'X_train_processed.csv' and 'X_test_processed.csv'.")
