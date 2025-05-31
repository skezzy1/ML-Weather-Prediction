
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

# 1. Завантаження даних
data = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Winter.csv')

# 2. Визначення фіч та цільової змінної
features = ['Year', 'Month', 'Average Temperature (celsius)', 
            'Max Temperature (celsius)', 'Min Temperature (celsius)', 
            'Average Wind Speed (m/s)', 'Max Snow Depth (cm)']
target = ['Total Precipitation (mm)']
y = data[target]
X = data[features]

# 3. Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Модель

model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)


model.fit(X_train, y_train.values.ravel())
print("=================================")
print("Lightgbm Модель натренована.")

# 5. Прогноз та оцінка
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=================================")
print("Lightgbm Результати:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")    
print(f"R² Score: {r2:.2f}")

# 6. Побудова графіку
plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', lw=2)
plt.title('Lightgbm: Прогнозовані проти Фактичних значень')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.tight_layout()
plt.show()
