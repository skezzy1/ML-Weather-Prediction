import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_preparation/csv/processed_final.csv')

features = ['Year', 'Month_sin', 'Month_cos', 'Average Temperature (celsius)', 
            'Max Temperature (celsius)', 'Min Temperature (celsius)', 
            'Average Wind Speed (m/s)', 'Max Snow Depth (cm)']
target = 'Total Precipitation (mm)'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)


rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print("=================================")
print("Random Forest Результати:")
print(f"MSE: {rf_mse:.2f}")
print(f"R² Score: {rf_r2:.2f}")

rf_importances = pd.Series(rf_model.feature_importances_, index=features)
print("\nRandom Forest Важливість функцій:")
print(rf_importances.sort_values(ascending=False))
print(rf_predictions)

scaler_y = StandardScaler()
y_full = data[['Total Precipitation (mm)']]
scaler_y.fit(y_full) 

y_test_inv = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
rf_predictions_inv = scaler_y.inverse_transform(rf_predictions.reshape(-1, 1)).flatten()

y_test_real = np.expm1(y_test_inv)  
rf_predictions_real = np.expm1(rf_predictions_inv)  

mse_real = mean_squared_error(y_test_real, rf_predictions_real)
mae_real = mean_absolute_error(y_test_real, rf_predictions_real)

plt.figure(figsize=(12, 6))
plt.scatter(y_test_real, rf_predictions_real, color='green')
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()],
         color='red', lw=2)
plt.title('Random Forest: Пронозовані проти Фактичних значень')
plt.xlabel('Фактичні значення')
plt.ylabel('Пронозовані значення')
plt.tight_layout()
plt.show()
