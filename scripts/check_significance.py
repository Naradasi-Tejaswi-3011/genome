import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the threshold for significance
P_VALUE_THRESHOLD = 0.005

# Load the dataset
file_path = "data/Copy of dataset.xlsx"
df = pd.read_excel(file_path)

# Create target variables based on p-value threshold
df['is_significant_CG'] = ((df['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                          (df['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
df['is_significant_CC'] = ((df['CC1_p_value'] < P_VALUE_THRESHOLD) & 
                          (df['CC2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
df['is_significant_CN'] = ((df['CN1_p_value'] < P_VALUE_THRESHOLD) & 
                          (df['CN2_p_value'] < P_VALUE_THRESHOLD)).astype(int)

# Check distribution of significant interactions for each condition
print(f"\nDistribution for Gemcitabine (CG) at p-value threshold {P_VALUE_THRESHOLD}:")
print(df['is_significant_CG'].value_counts())
print(f"Percentage significant: {df['is_significant_CG'].mean() * 100:.2f}%")

print(f"\nDistribution for Carboplatin (CC) at p-value threshold {P_VALUE_THRESHOLD}:")
print(df['is_significant_CC'].value_counts())
print(f"Percentage significant: {df['is_significant_CC'].mean() * 100:.2f}%")

print(f"\nDistribution for Normal (CN) at p-value threshold {P_VALUE_THRESHOLD}:")
print(df['is_significant_CN'].value_counts())
print(f"Percentage significant: {df['is_significant_CN'].mean() * 100:.2f}%")

# Check correlation between significance and IntGroup
print("\nRelationship between IntGroup and significance:")
for condition in ['CG', 'CC', 'CN']:
    print(f"\n{condition} - Distribution by IntGroup:")
    sig_col = f'is_significant_{condition}'
    print(pd.crosstab(df['IntGroup'], df[sig_col], normalize='index') * 100)

# Visualize the distribution for CG (Gemcitabine)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='is_significant_CG', data=df)
plt.title(f'Gemcitabine Significant vs Non-significant (p<{P_VALUE_THRESHOLD})')
plt.xlabel('Is Significant')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x='IntGroup', hue='is_significant_CG', data=df)
plt.title('Significance by Interaction Group (Gemcitabine)')
plt.xlabel('Interaction Group')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('significance_distribution.png')
print("\nVisualization saved as 'significance_distribution.png'")

# Check supporting pairs distribution
print("\nSupporting Pairs Statistics for Significant vs Non-significant interactions:")
for condition in ['CG', 'CC', 'CN']:
    sig_col = f'is_significant_{condition}'
    supp_col1 = f'{condition}1_SuppPairs' 
    supp_col2 = f'{condition}2_SuppPairs'
    
    print(f"\n{condition} Supporting Pairs for Significant Interactions:")
    print(df[df[sig_col] == 1][[supp_col1, supp_col2]].describe())
    
    print(f"\n{condition} Supporting Pairs for Non-significant Interactions:")
    print(df[df[sig_col] == 0][[supp_col1, supp_col2]].describe()) 