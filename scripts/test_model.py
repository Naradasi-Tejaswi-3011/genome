import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from genomic_classification import load_data, preprocess_data, feature_selection

# Function to predict significance of genomic interactions
def predict_significance(model, sample_data, numerical_cols, categorical_cols):
    """
    Predict whether genomic interactions are significant or not
    
    Parameters:
    -----------
    model : trained model object
        The RandomForest model (pipeline) trained to predict significance
    sample_data : DataFrame
        Data containing the features needed for prediction
    numerical_cols : list
        List of numerical feature column names
    categorical_cols : list
        List of categorical feature column names
        
    Returns:
    --------
    DataFrame with original data and predictions added
    """
    # Make a copy to avoid modifying the original
    data = sample_data.copy()
    
    # Create engineered features (same as during training)
    # Supporting pairs ratios
    data['CG_SuppPairs_Ratio'] = np.where(
        data['CG2_SuppPairs'] > 0,
        np.minimum(data['CG1_SuppPairs'] / data['CG2_SuppPairs'], 10),
        0
    )
    
    data['CC_SuppPairs_Ratio'] = np.where(
        data['CC2_SuppPairs'] > 0,
        np.minimum(data['CC1_SuppPairs'] / data['CC2_SuppPairs'], 10),
        0
    )
    
    data['CN_SuppPairs_Ratio'] = np.where(
        data['CN2_SuppPairs'] > 0,
        np.minimum(data['CN1_SuppPairs'] / data['CN2_SuppPairs'], 10),
        0
    )
    
    # Log distance
    data['log_distance'] = np.log10(data['distance'] + 1)
    
    # Clean any NaN or infinite values
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], 0)
            data[col] = data[col].fillna(0)
    
    # Extract only the features needed
    test_features = data[numerical_cols + categorical_cols]
    
    # Predict using the model
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    
    # Add predictions to the dataframe
    data['predicted_significant'] = predictions
    data['probability_significant'] = probabilities
    
    return data

def main():
    # Load the full dataset
    file_path = "data/Copy of dataset.xlsx"
    df = load_data(file_path)
    df_preprocessed = preprocess_data(df)
    
    # Since we don't have a separate test dataset, let's sample from our existing dataset
    # Random sample of 10 rows for testing (5 significant, 5 non-significant)
    significant_sample = df_preprocessed[df_preprocessed['is_significant'] == 1].sample(5, random_state=123)
    non_significant_sample = df_preprocessed[df_preprocessed['is_significant'] == 0].sample(5, random_state=123)
    test_sample = pd.concat([significant_sample, non_significant_sample])
    
    # Prepare features for prediction (same as in training)
    X, y, numerical_cols, categorical_cols, p_value_cols = feature_selection(df_preprocessed)
    
    # Load the model (assuming it's been saved)
    try:
        # Try to load previously saved model
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Loaded saved model")
    except FileNotFoundError:
        # If model file doesn't exist, we'll need to train it
        print("Model file not found. Running the genomic_classification.py script first.")
        from genomic_classification import main as train_main
        train_main()
        
        # Now try to save the model for future use
        import genomic_classification
        model = genomic_classification.train_random_forest(X, y, numerical_cols, categorical_cols, p_value_cols, df_preprocessed)
        with open('random_forest_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Trained and saved new model")
    
    # Make predictions on test samples
    results = predict_significance(model, test_sample, numerical_cols, categorical_cols)
    
    # Compare actual vs. predicted
    print("\nTest Results:")
    print("=" * 80)
    for idx, row in results.iterrows():
        print(f"Sample {idx}:")
        print(f"  InteractorID: {row['InteractorID']}")
        print(f"  Supp Pairs: CG1={row['CG1_SuppPairs']:.1f}, CG2={row['CG2_SuppPairs']:.1f}")
        print(f"  Distance: {row['distance']} bp")
        print(f"  P-values: CG1={row['CG1_p_value']:.6f}, CG2={row['CG2_p_value']:.6f}")
        print(f"  Interaction Type: {row['IntGroup']}")
        print(f"  Actual: {'Significant' if row['is_significant'] == 1 else 'Not Significant'}")
        print(f"  Predicted: {'Significant' if row['predicted_significant'] == 1 else 'Not Significant'}")
        print(f"  Prediction Probability: {row['probability_significant']:.4f}")
        print(f"  Correct Prediction: {'✓' if row['is_significant'] == row['predicted_significant'] else '✗'}")
        print("-" * 80)
    
    # Calculate accuracy on this test sample
    accuracy = (results['is_significant'] == results['predicted_significant']).mean()
    print(f"\nTest Sample Accuracy: {accuracy:.4f}")
    
    # Visualization of predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(results['CG1_SuppPairs'], results['CG2_SuppPairs'], 
                c=results['predicted_significant'], cmap='coolwarm', s=100, alpha=0.7)
    
    # Add markers around the actual significant points
    for idx, row in results.iterrows():
        if row['is_significant'] == 1:
            plt.scatter(row['CG1_SuppPairs'], row['CG2_SuppPairs'], 
                        s=160, facecolors='none', edgecolors='black', linewidths=2)
    
    plt.xlabel('CG1 Supporting Pairs')
    plt.ylabel('CG2 Supporting Pairs')
    plt.title('Predicted vs Actual Significant Interactions')
    plt.colorbar(label='Predicted Significance')
    plt.grid(alpha=0.3)
    plt.savefig('test_predictions.png')
    print("Visualization saved as 'test_predictions.png'")

if __name__ == "__main__":
    main() 