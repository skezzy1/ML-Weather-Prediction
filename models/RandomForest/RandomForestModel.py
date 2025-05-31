import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Winter.csv')

features = ['Year', 'Month', 'Average Temperature (celsius)', 
            'Max Temperature (celsius)', 'Min Temperature (celsius)',
            'Average Wind Speed (m/s)', 'Max Snow Depth (cm)']

target = ['Total Precipitation (mm)']

y = data[target]  
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(
    n_estimators=10000,
    max_depth=300,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train.values.ravel())
print("=================================")
print("Random Forest Модель натренована.")

rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("=================================")
print("Random Forest Результати:")
print(f"MSE: {rf_mse:.2f}")
print(f"MAE: {rf_mae:.2f}")
print(f"R² Score: {rf_r2:.2f}")

rf_importances = pd.Series(rf_model.feature_importances_, index=features)
print("\nRandom Forest Важливість функцій:")
print(rf_importances.sort_values(ascending=False))

plt.figure(figsize=(12, 6))
plt.scatter(y_test, rf_predictions, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', lw=2)
plt.title('Random Forest: Прогнозовані проти Фактичних значень')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.tight_layout()
plt.show()
