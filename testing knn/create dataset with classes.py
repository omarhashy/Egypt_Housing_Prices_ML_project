import pandas as pd

# Assuming the original data is in a CSV file named 'original_real_estate.csv'
# Read the original data
df = pd.read_csv('original_real_estate.csv')

# Create 30 quantile-based classes
num_classes = 30

# Use quantile-based binning to get the bin edges
price_bins = pd.qcut(df['price'], q=num_classes, retbins=True)[1]

# Create class names based on the price range
class_names = [f"{int(price_bins[i]):,} - {int(price_bins[i+1]):,}" for i in range(len(price_bins) - 1)]

# Assign each property to a class with named labels using quantile-based binning
df['price_class'] = pd.qcut(df['price'], q=num_classes, labels=class_names)

# Save the new DataFrame to a CSV file
df.to_csv('new_data_with_price_classes.csv', index=False)

print("New dataset with named price classes saved as 'new_data_with_price_classes.csv'.")