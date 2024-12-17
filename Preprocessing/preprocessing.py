import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'filepath.csv' 
data = pd.read_csv(file_path)

#drop rented apartments

initial_rows = len(data)
data = data[data['rent'] != 'yes']
rows_after_filter = len(data)
print(f"Rows before filtering: {initial_rows}, Rows after filtering rented apartments: {rows_after_filter}")


numerical_columns = ['price']


def plot_boxplot(data, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()


def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    print(Q1)
    print(Q3)
    print(IQR)
    lower_bound = max(200000,Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.7 * IQR
    print(lower_bound)
    print(upper_bound)

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Number of outliers in '{column}': {len(outliers)}")
    print("Sample outliers:")
    print(outliers.head())  
    # Remove outliers
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    print(f"Number of rows removed: {len(data) - len(filtered_data)}")
    return filtered_data


for col in numerical_columns:
    print(f"\nProcessing column: {col}")
    plot_boxplot(data, col)  # Visualize before removing outliers
    data = remove_outliers_iqr(data, col)  # Remove outliers
    plot_boxplot(data, col)  # Visualize after removing outliers


data.to_csv('cleaned_data.csv', index=False)
print("Cleaned data has been saved to 'cleaned_data.csv'.")
