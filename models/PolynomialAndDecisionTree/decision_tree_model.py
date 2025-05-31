import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Завантаження
df = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Winter.csv')

# Ознаки й ціль
features = ['Year', 'Month', 'Average Temperature (celsius)',
            'Average Wind Speed (m/s)', 'Max Snow Depth (cm)']
target = 'Total Precipitation (mm)'

X = df[features]
y = df[[target]]

# Масштабування
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled.ravel(), test_size=0.2, random_state=42)

# Дерево рішень
model = DecisionTreeRegressor(max_depth=8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Метрики
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🌳 Decision Tree Regressor")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Графік
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.6, label='Прогнози')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ідеал')
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.title("Decision Tree: Фактичні vs Прогнозовані")
plt.legend()
plt.tight_layout()
plt.show()
