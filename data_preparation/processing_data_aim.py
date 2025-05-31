import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

REGIONS = ("Center", "West", "North", "South", "East")
SEASONS = ("Spring", "Winter", "Summer", "Autumn")

temp_cols = ['Average Temperature (celsius)', 'Max Temperature (celsius)', 'Min Temperature (celsius)']
non_temp_cols = ['Average Wind Speed (m/s)', 'Max Snow Depth (cm)']
categorical_cols = ['City']
target_col = ['Total Precipitation (mm)']

for region in REGIONS:
    for season in SEASONS:
        input_path = f"data_preparation/csv/regions_seasons/{region}/{region}_{season}.csv"
        output_dir = f"data_preparation/csv/processed_regions/{region}"
        output_path = f"{output_dir}/{region}_{season}.csv"

        if not os.path.exists(input_path):
            print(f"⚠️ Файл не знайдено: {input_path}")
            continue

        print(f"\n🔄 Обробка: {input_path}")
        df = pd.read_csv(input_path)
        original_len = len(df)

        df = df.drop(columns=["Season", "Region"], errors='ignore')

        season_non_temp_cols = non_temp_cols.copy()
        if season == "Summer":
            season_non_temp_cols = [col for col in non_temp_cols if col != "Max Snow Depth (cm)"]
            df = df.drop(columns=["Max Snow Depth (cm)"], errors='ignore')

        X = df[temp_cols + season_non_temp_cols + categorical_cols]
        y = df[target_col]
        y_scaled = MinMaxScaler().fit_transform(y)

        preprocessor = ColumnTransformer(transformers=[
            ('temp', StandardScaler(), temp_cols),
            ('non_temp', MinMaxScaler(), season_non_temp_cols),
            ('city', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ])

        pipeline = Pipeline([('preprocess', preprocessor)])
        X_processed = pipeline.fit_transform(X)

        city_encoder = pipeline.named_steps['preprocess'].named_transformers_['city']
        city_columns = city_encoder.get_feature_names_out(['City'])
        final_columns = temp_cols + season_non_temp_cols + list(city_columns)
        df_final = pd.DataFrame(X_processed, columns=final_columns)

        df_final['Year'] = df['Year'].values
        df_final['Month'] = df['Month'].values
        df_final['Total Precipitation (mm)'] = y_scaled

        # Видалення викидів
        temp_mask = (df_final[temp_cols] >= -3) & (df_final[temp_cols] <= 3)
        temp_mask = temp_mask.all(axis=1)

        if season_non_temp_cols:
            non_temp_mask = (df_final[season_non_temp_cols] >= 0) & (df_final[season_non_temp_cols] <= 1)
            non_temp_mask = non_temp_mask.all(axis=1)
            mask = temp_mask & non_temp_mask
        else:
            mask = temp_mask

        df_filtered = df_final[mask]

        # Статистика
        total_rows = len(df_final)
        remaining_rows = len(df_filtered)
        removed_total = total_rows - remaining_rows
        removed_percent = removed_total / total_rows * 100

        removed_temp = total_rows - temp_mask.sum()
        removed_temp_percent = removed_temp / total_rows * 100

        if season_non_temp_cols:
            removed_non_temp = total_rows - non_temp_mask.sum()
            removed_non_temp_percent = removed_non_temp / total_rows * 100
        else:
            removed_non_temp = 0
            removed_non_temp_percent = 0.0

        print(f"📊 Статистика очищення:")
        print(f"🔸 Усього рядків до очищення: {total_rows}")
        print(f"🔹 Залишилось рядків після очищення: {remaining_rows}")
        print(f"❌ Видалено рядків: {removed_total} ({removed_percent:.2f}%)")
        print(f"  └ Викиди по StandardScaler (температура): {removed_temp} ({removed_temp_percent:.2f}%)")
        if season_non_temp_cols:
            print(f"  └ Викиди по MinMaxScaler (інші): {removed_non_temp} ({removed_non_temp_percent:.2f}%)")

        os.makedirs(output_dir, exist_ok=True)
        df_filtered.to_csv(output_path, index=False)
        print(f"✅ Збережено: {output_path}")
