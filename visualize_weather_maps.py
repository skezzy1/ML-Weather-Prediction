# visualize_weather_maps.py

import folium  
import pandas as pd  

# Load the data  
data = pd.read_csv('weather_data.csv')  

# Filter for Ukrainian cities  
ukrainian_cities = ['Kyiv', 'Lviv', 'Odesa', 'Kharkiv', 'Dnipro']  
data = data[data['city'].isin(ukrainian_cities)]  

# Create a map centered around Ukraine  
ukraine_map = folium.Map(location=[49.0, 32.0], zoom_start=5)  

# Add markers for each city  
for index, row in data.iterrows():  
    folium.Marker(  
        location=[row['latitude'], row['longitude']],  
        popup=f"{row['city']}: {row['temperature']}°C",  
        icon=folium.Icon(color='blue')  
    ).add_to(ukraine_map)  

# Save the map to an HTML file  
ukraine_map.save('ukrainian_weather_map.html')  
