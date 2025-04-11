import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

P_VALUE_THRESHOLD = 0.005


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')
file_path = os.path.join(data_dir, 'Copy of dataset.xlsx')


results_dir = os.path.join(os.path.dirname(current_dir), 'results')
os.makedirs(results_dir, exist_ok=True)

df = pd.read_excel(file_path)

df['is_significant_CG'] = ((df['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                          (df['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
df['is_significant_CC'] = ((df['CC1_p_value'] < P_VALUE_THRESHOLD) & 
                          (df['CC2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
df['is_significant_CN'] = ((df['CN1_p_value'] < P_VALUE_THRESHOLD) & 
                          (df['CN2_p_value'] < P_VALUE_THRESHOLD)).astype(int)


print(f"\nDistribution for Gemcitabine (CG) at p-value threshold {P_VALUE_THRESHOLD}:")
print(df['is_significant_CG'].value_counts())
print(f"Percentage significant: {df['is_significant_CG'].mean() * 100:.2f}%")

print(f"\nDistribution for Carboplatin (CC) at p-value threshold {P_VALUE_THRESHOLD}:")
print(df['is_significant_CC'].value_counts())
print(f"Percentage significant: {df['is_significant_CC'].mean() * 100:.2f}%")

print(f"\nDistribution for Normal (CN) at p-value threshold {P_VALUE_THRESHOLD}:")
print(df['is_significant_CN'].value_counts())
print(f"Percentage significant: {df['is_significant_CN'].mean() * 100:.2f}%")


print("\nRelationship between IntGroup and significance:")
for condition in ['CG', 'CC', 'CN']:
    print(f"\n{condition} - Distribution by IntGroup:")
    sig_col = f'is_significant_{condition}'
    print(pd.crosstab(df['IntGroup'], df[sig_col], normalize='index') * 100)


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
plt.savefig(os.path.join(results_dir, 'significance_distribution.png'))
print("\nVisualization saved as 'significance_distribution.png' in the results directory")

print("\nSupporting Pairs Statistics for Significant vs Non-significant interactions:")
for condition in ['CG', 'CC', 'CN']:
    sig_col = f'is_significant_{condition}'
    supp_col1 = f'{condition}1_SuppPairs' 
    supp_col2 = f'{condition}2_SuppPairs'
    
    print(f"\n{condition} Supporting Pairs for Significant Interactions:")
    print(df[df[sig_col] == 1][[supp_col1, supp_col2]].describe())
    
    print(f"\n{condition} Supporting Pairs for Non-significant Interactions:")
    print(df[df[sig_col] == 0][[supp_col1, supp_col2]].describe()) 