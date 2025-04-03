import pandas as pd

# Load the full dataset
file_path = "data/Copy of dataset.xlsx"
df = pd.read_excel(file_path)

# Create a small sample for testing (20 random rows)
sample = df.sample(20, random_state=42)

# Save to a new file
sample.to_excel("sample_test_data.xlsx", index=False)
print(f"Created sample test data with {len(sample)} rows")
print("Saved as 'sample_test_data.xlsx'") 