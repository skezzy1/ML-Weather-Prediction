import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

FILE_PATH = "data_preparation/csv/data.csv"
SAVE_PATH = "data_preparation/csv/processed_final.csv"
VISUAL_PATH = "data_preparation/visuals/"

NUMERIC_COLUMNS = [
    "Average Temperature (celsius)",
    "Max Temperature (celsius)",
    "Min Temperature (celsius)",
    "Average Wind Speed (m/s)",
    "Total Precipitation (mm)",
    "Max Snow Depth (cm)"
]

NON_NEGATIVE_COLS = [
    "Average Wind Speed (m/s)",
    "Total Precipitation (mm)",
    "Max Snow Depth (cm)"
]

LOG_COLS = [
    "Total Precipitation (mm)",
    "Max Snow Depth (cm)"
]

def make_filename_safe(col):
    return col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

def preprocess_and_save(df):
    df = df.copy()
    for col in NON_NEGATIVE_COLS:
        df[col] = df[col].clip(lower=0)
    total_removed = 0
    for col in NUMERIC_COLUMNS:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        before = len(df)
        df = df[(z < 3) & (z > -3)]
        after = len(df)
        removed = before - after
        print(f"🔍 Видалено з {col}: {removed} рядків ({(removed / before) * 100:.2f}%)")
        total_removed += removed
    print(f"\n🔍 Загалом видалено {total_removed} рядків з екстремальними Z-оцінками")

    for col in LOG_COLS:
        df[col] = np.log1p(df[col])
        print(f"🔁 Логарифмовано: {col}")

    if "Month" in df.columns:
        df["Month"] = df["Month"].clip(1, 12)
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        
        df["Month_sin"] = df["Month_sin"].apply(lambda x: 0 if abs(x) < 1e-15 else x)

        print("📆 Циклічне кодування Month застосовано")
    if "City" in df.columns:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        city_encoded = ohe.fit_transform(df[["City"]])
        city_cols = ohe.get_feature_names_out(["City"])
        df_city = pd.DataFrame(city_encoded, columns=city_cols, index=df.index)
        df = pd.concat([df.drop(columns=["City"]), df_city], axis=1)
        print("🏙️ OneHotEncoding застосовано до City")

    scaler = StandardScaler()
    df[NUMERIC_COLUMNS] = scaler.fit_transform(df[NUMERIC_COLUMNS])
    print("⚖️ Стандартизація числових завершена")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ Дані збережено у: {SAVE_PATH}")
    return df

if __name__ == "__main__":
    print("📥 Завантаження...")
    df = pd.read_csv(FILE_PATH)
    os.makedirs(VISUAL_PATH, exist_ok=True)
    df_clean = preprocess_and_save(df)
