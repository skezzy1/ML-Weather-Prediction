import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('data_preparation/data.csv')

le = LabelEncoder()
data['City_encoded'] = le.fit_transform(data['City'])

features = ['City_encoded', 'Year', 'Month', 'Average Temperature (celsius)', 
            'Max Temperature (celsius)', 'Min Temperature (celsius)', 
            'Average Wind Speed (m/s)', 'Total Precipitation (mm)']
target = 'Max Snow Depth (cm)'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\nXGBoost Results:")
print(f"MSE: {xgb_mse:.2f}")
print(f"R² Score: {xgb_r2:.2f}")

xgb_importances = pd.Series(xgb_model.feature_importances_, index=features)

print("\nXGBoost Feature Importances:")
print(xgb_importances.sort_values(ascending=False))


plt.figure(figsize=(12, 6))
plt.scatter(y_test, xgb_predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('XGBoost: Predictions vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.tight_layout()
plt.show()
