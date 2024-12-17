import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'data.csv'  
data = pd.read_csv(file_path)

# Handling Missing Values


def check_missing_values(data):
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    print("Missing Values in Columns:")
    print(missing)
    return missing


def handle_missing_values(data):
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype == 'object':  # Categorical columns
                mode_value = data[column].mode()[0]
                data[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in '{column}' with mode: {mode_value}")
            else:  # Numerical columns
                median_value = data[column].median()
                data[column].fillna(median_value, inplace=True)
                print(f"Filled missing values in '{column}' with median: {median_value}")
    return data

print("\nChecking for missing values...")
missing_values_before = check_missing_values(data)

print("\nHandling missing values...")
data = handle_missing_values(data)

print("\nChecking missing values after handling...")
missing_values_after = check_missing_values(data)
if missing_values_after.empty:
    print("No missing values remaining!")

data.to_csv('data_no_missing.csv', index=False)
print("\nData with missing values handled saved as 'data_no_missing.csv'.")
