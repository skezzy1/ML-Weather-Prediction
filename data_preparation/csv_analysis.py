import pandas as pd 

dataset = pd.read_csv("data_preparation/csv/data_with_regions.csv")

X = dataset.iloc[: , :].values
y = dataset.iloc[: , 0].values

#print(X)
unique_data_meaning = set(y) # Unique cities in dataset
print(unique_data_meaning)
print("==================================================================")
# Check for missing values
print(dataset.isnull().sum())
print("==================================================================")
# Check percentage of missing values per column
print(dataset.isnull().mean() * 100)
print("==================================================================")
duplicates = dataset.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

#dataset['Average Wind Speed (m/s)'].fillna(dataset['Average Wind Speed (m/s)'].mean(), inplace=True)
