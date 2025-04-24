import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

dataset = pd.read_csv("data_preparation/csv/data_with_regions.csv")


categorical_columns = dataset.select_dtypes(include=['object']).columns
numerical_columns = dataset.select_dtypes(exclude=['object']).columns

numerical_imputer = SimpleImputer(strategy='mean')
dataset[numerical_columns] = numerical_imputer.fit_transform(dataset[numerical_columns])

categorical_imputer = SimpleImputer(strategy='most_frequent')
dataset[categorical_columns] = categorical_imputer.fit_transform(dataset[categorical_columns])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False), categorical_columns)], 
    remainder='passthrough'
)

X = dataset.drop(['Total Precipitation (mm)'], axis='columns')
y = dataset["Total Precipitation (mm)"].values

X_transformed = ct.fit_transform(X)

encoded_columns = ct.transformers_[0][1].get_feature_names_out(categorical_columns)

print("Shape of transformed data:", X_transformed.shape)

print("Number of encoded columns:", len(encoded_columns))

for col in categorical_columns:
    print(f"Unique values in {col}: {dataset[col].unique()}")

all_columns = list(encoded_columns) + list(numerical_columns)

if len(all_columns) != X_transformed.shape[1]:
    print(f"Mismatch! {len(all_columns)} columns, but {X_transformed.shape[1]} columns in transformed data.")
else:
    X_df = pd.DataFrame(X_transformed, columns=all_columns)

    selector = SelectKBest(f_classif, k='all')
    X_new = selector.fit_transform(X_df, y)

    print("Feature scores from ANOVA F-test:")
    print(selector.scores_)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

    pd.DataFrame(X_train).to_csv("data_preparation/X_train_processed.csv", index=False)
    pd.DataFrame(X_test).to_csv("data_preparation/X_test_processed.csv", index=False)
    pd.DataFrame(y_train).to_csv("data_preparation/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data_preparation/y_test.csv", index=False)

    print("Data preparation completed. Processed files saved in the 'data_preparation' folder.")
