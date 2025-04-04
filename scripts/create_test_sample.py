import pandas as pd

file_path = "data/Copy of dataset.xlsx"
df = pd.read_excel(file_path)

sample = df.sample(20, random_state=42)

sample.to_excel("sample_test_data.xlsx", index=False)
print(f"Created sample test data with {len(sample)} rows")
print("Saved as 'sample_test_data.xlsx'") 