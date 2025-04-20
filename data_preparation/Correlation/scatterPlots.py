import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset = pd.read_csv('data_preparation/csv/data.csv', encoding='utf-8')

sns.histplot(dataset["Total Precipitation (mm)"], bins=50, kde=True)
plt.title("Distribution of Total Precipitation")
plt.show()
dataset["clipped"] = dataset["Max Snow Depth (cm)"].clip(upper=100)  # якщо викиди > 100
dataset["log_precip"] = np.log1p(dataset["Total Precipitation (mm)"])
