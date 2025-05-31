
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# 1. Завантаження даних
data = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Summer.csv')

# 2. Визначення фіч та цільової змінної
features = ['Year', 'Month', 'Average Temperature (celsius)', 
            'Max Temperature (celsius)', 'Min Temperature (celsius)',
            'Average Wind Speed (m/s)']
target = ['Total Precipitation (mm)']
y = data[target]
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=5)


model.fit(X_train, y_train.values.ravel())
print("=================================")
print("Knn Модель натренована.")

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=================================")
print("Knn Результати:")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', lw=2)
plt.title('Knn: Прогнозовані проти Фактичних значень')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.tight_layout()
plt.show()
