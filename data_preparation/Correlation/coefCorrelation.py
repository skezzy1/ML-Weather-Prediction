import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#df = pd.read_csv("weather_data.csv")
df = pd.read_csv("data_preparation/csv/cities/lviv_data.csv")
#df = pd.read_csv("data_preparation/austin_weather.csv")
df.columns = [col.strip() for col in df.columns]

'''
selected_columns = [
    "Temperature_C",  
    "Humidity_pct",
    "Wind_Speed_kmh",
    "Precipitation_mm"
]
'''
city_name = "Львів"
city_data = df[df['City'] == city_name]
selected_columns = [
    "Average Temperature (celsius)",
    "Max Temperature (celsius)",
    "Min Temperature (celsius)",
    "Average Wind Speed (m/s)",
    "Total Precipitation (mm)",
    "Max Snow Depth (cm)"
]
'''
selected_columns = [
    "TempHighF",
    "TempAvgF",
    "TempLowF",
    "DewPointHighF",
    "DewPointAvgF",
    "DewPointLowF",
    "HumidityHighPercent",
    "HumidityAvgPercent",
    "HumidityLowPercent",
    "SeaLevelPressureHighInches",
    "SeaLevelPressureAvgInches",
    "SeaLevelPressureLowInches",
    "VisibilityHighMiles",
    "VisibilityAvgMiles",
    "VisibilityLowMiles",
    "WindHighMPH",
    "WindAvgMPH",
    "WindGustMPH",
    "PrecipitationSumInches"

]
'''
df_selected = df[selected_columns]
#df_selected = df[selected_columns].replace({'-': np.nan, 'T' : np.nan})

correlation_matrix = df_selected.corr()

plt.figure(figsize=(17, 15))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title(f'Кореляційна матриця погодних параметрів для {city_name}')
plt.show()

print(correlation_matrix)
city_data.to_csv('data_preparation/csv/cities/lviv_data.csv', index=False, encoding='utf-8')
correltion_result = f"data_preparation/Correlation/correlation_result_{city_name}.txt"
with open(correltion_result,  "w", encoding="utf-8") as file:
    file.write(correlation_matrix.to_string())