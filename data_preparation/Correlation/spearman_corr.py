# Re-import needed modules after code execution reset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reload dataset
df = pd.read_csv("data_preparation/csv/data_with_regions.csv")
df.columns = [col.strip() for col in df.columns]

# Filter by city
#city_name = "Львів"
#city_data = df[df['City'] == city_name]

# Define columns to include
selected_columns = [
    "Average Temperature (celsius)",
    "Max Temperature (celsius)",
    "Min Temperature (celsius)",
    "Average Wind Speed (m/s)",
    "Total Precipitation (mm)",
    "Max Snow Depth (cm)"
]

# Calculate Spearman's rank correlation
spearman_corr = df[selected_columns].corr(method="spearman")

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title(f"Spearman Rank Correlation Matrix")
plt.tight_layout()
plt.show()

