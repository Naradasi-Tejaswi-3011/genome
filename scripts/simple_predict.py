import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def load_model():
    """Load the trained model and metadata"""
    try:
        # Update paths to use models directory
        with open('../models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def preprocess_data(data, feature_columns):
    """Preprocess the input data"""
    try:
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data[feature_columns])
        return X
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return None

def test_model():
    """Test the model on sample data"""
    print("Loading model and metadata...")
    model, metadata = load_model()
    if model is None or metadata is None:
        return
    
    # Create test data with known values
    test_data = pd.DataFrame({
        'Distance': [1000000, 2000000, 500000],
        'GC_Content': [0.45, 0.38, 0.52],
        'Repeat_Content': [0.2, 0.15, 0.25],
        'Conservation_Score': [0.8, 0.6, 0.9],
        'Expression_Level': [5.2, 3.8, 6.5],
        'Chromatin_State': [1, 2, 1],
        'TF_Binding_Sites': [3, 2, 4],
        'Histone_Marks': [2, 1, 3],
        'DNA_Methylation': [0.3, 0.4, 0.2],
        'Chromosome_Length': [250000000, 200000000, 300000000]
    })
    
    # Preprocess data
    X = preprocess_data(test_data, metadata['feature_columns'])
    if X is None:
        return
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Print results
    print("\nTest Results:")
    print("=" * 80)
    for i, (idx, row) in enumerate(test_data.iterrows()):
        print(f"\nSample {i+1}:")
        print(f"  Distance: {row['Distance']} bp")
        print(f"  GC Content: {row['GC_Content']:.2f}")
        print(f"  Expression Level: {row['Expression_Level']:.2f}")
        print(f"  Predicted Interaction: {'Yes' if predictions[i] == 1 else 'No'}")
        print(f"  Interaction Probability: {probabilities[i][1]:.4f}")
        print("-" * 80)
    
    # Save results
    test_data['Predicted_Interaction'] = predictions
    test_data['Interaction_Probability'] = probabilities[:, 1]
    test_data.to_excel('../results/test_predictions.xlsx', index=False)
    print("\nResults saved to ../results/test_predictions.xlsx")

if __name__ == "__main__":
    test_model()

# Load the model and metadata
print("Loading model and metadata...")
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    numerical_cols = metadata['numerical_cols']
    categorical_cols = metadata['categorical_cols']

# Load sample test data
print("Loading test data...")
test_data = pd.read_excel('sample_test_data.xlsx')
print(f"Loaded {len(test_data)} test samples")

# Function to predict significance
def predict_significance(data):
    """Prepare data and predict significance"""
    # Create a copy of the data
    data_copy = data.copy()
    
    # Create engineered features
    data_copy['CG_SuppPairs_Ratio'] = np.where(
        data_copy['CG2_SuppPairs'] > 0,
        np.minimum(data_copy['CG1_SuppPairs'] / data_copy['CG2_SuppPairs'], 10),
        0
    )
    
    data_copy['CC_SuppPairs_Ratio'] = np.where(
        data_copy['CC2_SuppPairs'] > 0,
        np.minimum(data_copy['CC1_SuppPairs'] / data_copy['CC2_SuppPairs'], 10),
        0
    )
    
    data_copy['CN_SuppPairs_Ratio'] = np.where(
        data_copy['CN2_SuppPairs'] > 0,
        np.minimum(data_copy['CN1_SuppPairs'] / data_copy['CN2_SuppPairs'], 10),
        0
    )
    
    data_copy['log_distance'] = np.log10(data_copy['distance'] + 1)
    
    # Handle missing or infinite values
    for col in numerical_cols:
        if col in data_copy.columns:
            data_copy[col] = data_copy[col].replace([np.inf, -np.inf], 0)
            data_copy[col] = data_copy[col].fillna(0)
    
    # Select features for prediction
    features = data_copy[numerical_cols + categorical_cols]
    
    # Make predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    
    # Add predictions to data
    data_copy['predicted_significant'] = predictions
    data_copy['probability_significant'] = probabilities
    
    return data_copy

# Make predictions
print("Predicting significance...")
results = predict_significance(test_data)

# Calculate actual significance based on p-value threshold (0.005)
P_VALUE_THRESHOLD = 0.005
results['actual_significant'] = ((results['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                               (results['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)

# Print results
print("\nPrediction Results:")
print("=" * 80)
for i, (idx, row) in enumerate(results.iterrows()):
    print(f"Sample {i+1}:")
    print(f"  InteractorID: {row['InteractorID']}")
    print(f"  Supp Pairs: CG1={row['CG1_SuppPairs']:.1f}, CG2={row['CG2_SuppPairs']:.1f}")
    print(f"  Distance: {row['distance']} bp")
    print(f"  P-values: CG1={row['CG1_p_value']:.6f}, CG2={row['CG2_p_value']:.6f}")
    print(f"  Interaction Type: {row['IntGroup']}")
    print(f"  Actual: {'Significant' if row['actual_significant'] == 1 else 'Not Significant'}")
    print(f"  Predicted: {'Significant' if row['predicted_significant'] == 1 else 'Not Significant'}")
    print(f"  Prediction Probability: {row['probability_significant']:.4f}")
    print(f"  Correct Prediction: {'✓' if row['actual_significant'] == row['predicted_significant'] else '✗'}")
    print("-" * 80)

# Print summary statistics
accuracy = (results['actual_significant'] == results['predicted_significant']).mean()
sig_count = results['predicted_significant'].sum()
total = len(results)

print(f"\nSummary:")
print(f"Predicted {sig_count} significant interactions out of {total} ({sig_count/total:.1%})")
print(f"Model accuracy on this test set: {accuracy:.2%}")

# Generate confusion matrix
cm = confusion_matrix(results['actual_significant'], results['predicted_significant'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('sample_test_confusion_matrix.png')

# Print classification report
print("\nClassification Report:")
print(classification_report(results['actual_significant'], results['predicted_significant']))

print("\nAnalysis complete! Confusion matrix saved to 'sample_test_confusion_matrix.png'")

# Save results to Excel
results.to_excel('sample_test_results.xlsx', index=False)
print("Detailed results saved to 'sample_test_results.xlsx'") 