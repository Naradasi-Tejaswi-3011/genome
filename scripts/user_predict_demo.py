import pandas as pd
import numpy as np
import pickle
import time

# Load the model and metadata
print("Loading model and metadata...")
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    numerical_cols = metadata['numerical_cols']
    categorical_cols = metadata['categorical_cols']

# Define sample interaction values for demonstration
sample_interactions = [
    {
        "CG1_SuppPairs": 25,
        "CG2_SuppPairs": 30,
        "CC1_SuppPairs": 15,
        "CC2_SuppPairs": 18,
        "CN1_SuppPairs": 12,
        "CN2_SuppPairs": 14,
        "distance": 150000,
        "IntGroup": "PD",
        "Strand": "+",
        "NofInts": 3,
        "Annotation": 1,
        "InteractorAnnotation": 1
    },
    {
        "CG1_SuppPairs": 8,
        "CG2_SuppPairs": 5,
        "CC1_SuppPairs": 7,
        "CC2_SuppPairs": 6,
        "CN1_SuppPairs": 9,
        "CN2_SuppPairs": 4,
        "distance": 55000,
        "IntGroup": "PP",
        "Strand": "-",
        "NofInts": 2,
        "Annotation": 1,
        "InteractorAnnotation": 1
    },
    {
        "CG1_SuppPairs": 60,
        "CG2_SuppPairs": 45,
        "CC1_SuppPairs": 30,
        "CC2_SuppPairs": 28,
        "CN1_SuppPairs": 25,
        "CN2_SuppPairs": 20,
        "distance": 80000,
        "IntGroup": "PD",
        "Strand": "+",
        "NofInts": 3,
        "Annotation": 1,
        "InteractorAnnotation": 1
    }
]

def prepare_and_predict(interaction_data):
    """Process interaction data and predict significance"""
    # Convert to DataFrame
    df = pd.DataFrame([interaction_data])
    
    # Create engineered features
    df['CG_SuppPairs_Ratio'] = np.where(
        df['CG2_SuppPairs'] > 0,
        np.minimum(df['CG1_SuppPairs'] / df['CG2_SuppPairs'], 10),
        0
    )
    
    df['CC_SuppPairs_Ratio'] = np.where(
        df['CC2_SuppPairs'] > 0,
        np.minimum(df['CC1_SuppPairs'] / df['CC2_SuppPairs'], 10),
        0
    )
    
    df['CN_SuppPairs_Ratio'] = np.where(
        df['CN2_SuppPairs'] > 0,
        np.minimum(df['CN1_SuppPairs'] / df['CN2_SuppPairs'], 10),
        0
    )
    
    df['log_distance'] = np.log10(df['distance'] + 1)
    
    # Handle missing or infinite values
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].fillna(0)
    
    # Select features for prediction
    features = df[numerical_cols + categorical_cols]
    
    # Make predictions
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]
    
    # Display result
    print(f"\n=== Prediction Result ===")
    print(f"Prediction: {'Significant' if prediction == 1 else 'Not Significant'}")
    print(f"Confidence: {probability:.2%}")
    
    return prediction, probability

def simulate_user_input():
    """Simulate user entering input values with provided examples"""
    print("\n=== Genomic Interaction Significance Predictor ===")
    print("(Demo version with provided input values)")
    
    for i, interaction in enumerate(sample_interactions):
        print(f"\n\n----------- Example {i+1} -----------")
        print("\nEnter the following details to predict significance:")
        
        # Simulate entering each value with a short delay
        print(f"\nEnter CG1 Supporting Pairs: {interaction['CG1_SuppPairs']}")
        time.sleep(0.5)
        
        print(f"Enter CG2 Supporting Pairs: {interaction['CG2_SuppPairs']}")
        time.sleep(0.5)
        
        print(f"Enter CC1 Supporting Pairs: {interaction['CC1_SuppPairs']}")
        time.sleep(0.5)
        
        print(f"Enter CC2 Supporting Pairs: {interaction['CC2_SuppPairs']}")
        time.sleep(0.5)
        
        print(f"Enter CN1 Supporting Pairs: {interaction['CN1_SuppPairs']}")
        time.sleep(0.5)
        
        print(f"Enter CN2 Supporting Pairs: {interaction['CN2_SuppPairs']}")
        time.sleep(0.5)
        
        print(f"\nEnter genomic distance (bp): {interaction['distance']}")
        time.sleep(0.5)
        
        print(f"\nEnter interaction type (PP, PD, or DD): {interaction['IntGroup']}")
        time.sleep(0.5)
        
        print(f"\nEnter strand (+ or -): {interaction['Strand']}")
        time.sleep(0.5)
        
        print(f"\nEnter number of interactions: {interaction['NofInts']}")
        time.sleep(1)
        
        # Make prediction
        prepare_and_predict(interaction)
        
        # Ask to continue
        if i < len(sample_interactions) - 1:
            print("\nProcessing next example...")
            time.sleep(1)
    
    print("\nDemo complete! These examples show how the model predicts significant")
    print("and non-significant genomic interactions based on input parameters.")

if __name__ == "__main__":
    simulate_user_input() 