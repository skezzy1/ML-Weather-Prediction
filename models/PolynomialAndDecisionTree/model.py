# Re-import after code execution state reset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
df = pd.read_csv('data_preparation/csv/processed_regions/Center/Center_Winter.csv')

# Вибір ознак і цільової змінної
features = ['Year', 'Month', 'Average Temperature (celsius)', 
            'Average Wind Speed (m/s)', 'Max Snow Depth (cm)']
target = 'Total Precipitation (mm)'

X = df[features]
y = df[target]

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Поліноміальна регресія
poly_pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
y_poly_pred = poly_pipeline.predict(X_test)

# Дерево рішень
tree_model = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_model.fit(X_train, y_train)
y_tree_pred = tree_model.predict(X_test)

# Оцінка моделей
poly_mse = mean_squared_error(y_test, y_poly_pred)
poly_r2 = r2_score(y_test, y_poly_pred)

tree_mse = mean_squared_error(y_test, y_tree_pred)
tree_r2 = r2_score(y_test, y_tree_pred)

(poly_mse, poly_r2), (tree_mse, tree_r2)
