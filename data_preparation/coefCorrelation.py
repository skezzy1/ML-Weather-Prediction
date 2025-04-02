import pandas as pd

# Завантаження даних
df = pd.read_csv("data_preparation/data.csv")

# Перейменування стовпців для зручності
df.columns = [col.strip() for col in df.columns]
'''
selected_columns = [
    "Temperature_C",  
    "Humidity_pct",
    "Wind_Speed_kmh",
    "Precipitation_mm"
]'
'''
# Вибір конкретних змінних для кореляції
selected_columns = [
    "Total Precipitation (mm)",  
    "Max Temperature (celsius)",
    "Max Snow Depth (cm)",
    "Average Temperature (celsius)"
]

df_selected = df[selected_columns]

# Обчислення кореляцій між вибраними змінними
correlation_matrix = df_selected.corr()

# Виведення матриці кореляцій
print(correlation_matrix)
