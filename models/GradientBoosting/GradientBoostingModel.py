import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# 1. Завантаження даних
data = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Winter.csv')

# 2. Формуємо фічі
base_features = ['Year', 'Month', 'Average Temperature (celsius)', 
                 'Average Wind Speed (m/s)', 'Max Temperature (celsius)', 'Min Temperature (celsius)', 'Max Snow Depth (cm)']

# Знаходимо всі колонки, які є one-hot для міст
city_features = [col for col in data.columns if col.startswith('City_')]

# Об'єднуємо всі фічі
features = base_features + city_features
target = ['Total Precipitation (mm)']

X = data[features]
y = data[target]

# 3. Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Модель
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=15, random_state=42)
model.fit(X_train, y_train.values.ravel())
print("=================================")
print("Gradient Boosting Модель натренована.")

# 5. Прогноз та оцінка
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=================================")
print("Gradient Boosting Результати:")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 6. Побудова графіку
plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions, color='green', alpha=0.6)
plt.plot([y_test.min().values[0], y_test.max().values[0]],
         [y_test.min().values[0], y_test.max().values[0]],
         color='red', lw=2)
plt.title('Gradient Boosting: Прогнозовані проти Фактичних значень')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.tight_layout()
plt.show()
