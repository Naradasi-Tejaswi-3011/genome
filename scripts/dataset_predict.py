import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import os

def main():
    print("====== GENOMIC INTERACTION MODEL EVALUATION ======")
    
    # Load the model
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create test dataset with same structure as training data
    test_data = pd.DataFrame({
        'IntGroup': ['PP'] * 100,  # Using PP (Promoter-Promoter) as documented
        'Strand': ['+'] * 100,
        'distance': np.random.randint(1000, 1000000, 100),
        'CG1_SuppPairs': np.random.randint(0, 100, 100),
        'CG2_SuppPairs': np.random.randint(0, 100, 100),
        'CC1_SuppPairs': np.random.randint(0, 100, 100),
        'CC2_SuppPairs': np.random.randint(0, 100, 100),
        'CN1_SuppPairs': np.random.randint(0, 100, 100),
        'CN2_SuppPairs': np.random.randint(0, 100, 100),
        'NofInts': np.random.randint(1, 10, 100),
        'Annotation': np.random.randint(0, 3, 100),
        'InteractorAnnotation': np.random.randint(0, 3, 100),
        'Normal': np.random.choice([0, 1], 100),
        'CarboplatinTreated': np.random.choice([0, 1], 100),
        'GemcitabineTreated': np.random.choice([0, 1], 100)
    })

    # Add engineered features
    test_data['CG_SuppPairs_Ratio'] = test_data.apply(
        lambda row: row['CG1_SuppPairs'] / row['CG2_SuppPairs'] if row['CG2_SuppPairs'] > 0 else 0, axis=1)
    test_data['CC_SuppPairs_Ratio'] = test_data.apply(
        lambda row: row['CC1_SuppPairs'] / row['CC2_SuppPairs'] if row['CC2_SuppPairs'] > 0 else 0, axis=1)
    test_data['CN_SuppPairs_Ratio'] = test_data.apply(
        lambda row: row['CN1_SuppPairs'] / row['CN2_SuppPairs'] if row['CN2_SuppPairs'] > 0 else 0, axis=1)
    test_data['log_distance'] = np.log10(test_data['distance'] + 1)

    # Save test data
    os.makedirs('data', exist_ok=True)
    test_data.to_excel('data/test_data.xlsx', index=False)
    print(f"Test data saved to data/test_data.xlsx with {len(test_data)} samples")

    # Make predictions
    print("\nMaking predictions...")
    try:
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)[:, 1]
        
        # Add predictions to results
        test_data['Predicted_Interaction'] = predictions
        test_data['Interaction_Probability'] = probabilities
        
        # Save results
        os.makedirs('results', exist_ok=True)
        test_data.to_excel('results/prediction_results.xlsx', index=False)
        
        # Print summary
        print(f"\nPrediction Results:")
        print(f"Total samples: {len(test_data)}")
        print(f"Predicted interactions: {sum(predictions)}")
        print(f"Predicted non-interactions: {len(predictions) - sum(predictions)}")
        print("\nResults saved to results/prediction_results.xlsx")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main() 