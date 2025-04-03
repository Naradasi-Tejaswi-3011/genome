import pandas as pd

# Load the dataset
file_path = "data/Copy of dataset.xlsx"
df = pd.read_excel(file_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
for col in df.columns:
    print(f"- {col}")

print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Get info about data types
print("\nData types:")
print(df.dtypes)

# Check for unique values in categorical columns
print("\nSample of potential categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f"\n{col} - Unique values: {len(unique_values)}")
    print(unique_values[:10])  # Print first 10 unique values 