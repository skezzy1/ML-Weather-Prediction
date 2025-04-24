from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

df = pd.read_csv("data_preparation/csv/processed_final.csv")


df_numeric = df.select_dtypes(include=["number"])

df_numeric = df_numeric.dropna() 
df_numeric = df_numeric[~df_numeric.isin([float('inf'), -float('inf')]).any(axis=1)] 

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = df_numeric.columns
vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

print(vif_data)
