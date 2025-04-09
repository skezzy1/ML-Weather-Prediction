from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

df = pd.read_csv("data_preparation/data.csv")

# Select numeric columns
df_numeric = df.select_dtypes(include=["number"])

# Check for NaN or infinite values and drop rows or fill them
df_numeric = df_numeric.dropna()  # Drop rows with NaN values
df_numeric = df_numeric[~df_numeric.isin([float('inf'), -float('inf')]).any(axis=1)]  # Drop rows with infinite values

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = df_numeric.columns
vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

print(vif_data)
