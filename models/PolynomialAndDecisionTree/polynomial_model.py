import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
df = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Winter.csv')

# Вибір ознак і цільової змінної
features = ['Year', 'Month', 'Average Temperature (celsius)',
            'Max Temperature (celsius)', 'Min Temperature (celsius)',
            'Average Wind Speed (m/s)', 'Max Snow Depth (cm)']
target = 'Total Precipitation (mm)'

X = df[features]
y = df[target]

# Розбиття
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Поліноміальна регресія з масштабуванням
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

poly_pipeline.fit(X_train, y_train)
y_pred = poly_pipeline.predict(X_test)

# Метрики
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Результати
print("🔢 Polynomial Regression")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Графік
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Прогнози')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ідеал')
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.title("Polynomial Regression: Фактичні vs Прогнозовані")
plt.legend()
plt.tight_layout()
plt.show()
