import pandas as pd
import numpy as np
import pickle

# Load the model and metadata
print("Loading model and metadata...")
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        numerical_cols = metadata['numerical_cols']
        categorical_cols = metadata['categorical_cols']
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Run genomic_classification.py first.")
    exit()

def predict_single_interaction():
    """Take user input for a single genomic interaction and predict significance"""
    print("\n=== Genomic Interaction Significance Predictor ===")
    print("Enter the following details to predict significance:")
    
    # Create an empty dictionary to store user inputs
    interaction = {}
    
    # Get input for essential features
    try:
        # Supporting pairs
        interaction['CG1_SuppPairs'] = float(input("\nEnter CG1 Supporting Pairs: "))
        interaction['CG2_SuppPairs'] = float(input("Enter CG2 Supporting Pairs: "))
        interaction['CC1_SuppPairs'] = float(input("Enter CC1 Supporting Pairs: "))
        interaction['CC2_SuppPairs'] = float(input("Enter CC2 Supporting Pairs: "))
        interaction['CN1_SuppPairs'] = float(input("Enter CN1 Supporting Pairs: "))
        interaction['CN2_SuppPairs'] = float(input("Enter CN2 Supporting Pairs: "))
        
        # Distance
        interaction['distance'] = float(input("\nEnter genomic distance (bp): "))
        
        # Interaction type
        int_type = input("\nEnter interaction type (PP, PD, or DD): ").upper()
        while int_type not in ['PP', 'PD', 'DD']:
            print("Invalid input. Please enter PP, PD, or DD.")
            int_type = input("Enter interaction type (PP, PD, or DD): ").upper()
        interaction['IntGroup'] = int_type
        
        # Strand
        strand = input("\nEnter strand (+ or -): ")
        while strand not in ['+', '-']:
            print("Invalid input. Please enter + or -.")
            strand = input("Enter strand (+ or -): ")
        interaction['Strand'] = strand
        
        # Number of interactions
        interaction['NofInts'] = float(input("\nEnter number of interactions: "))
        
        # Default values for other required columns
        interaction['Annotation'] = 1  # Default value
        interaction['InteractorAnnotation'] = 1  # Default value
        
    except ValueError:
        print("Error: Please enter numeric values for numbers.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame([interaction])
    
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
    print("\n=== Prediction Result ===")
    print(f"Prediction: {'Significant' if prediction == 1 else 'Not Significant'}")
    print(f"Confidence: {probability:.2%}")
    
    return df, prediction, probability

def main():
    while True:
        result = predict_single_interaction()
        if result is None:
            print("Prediction failed. Please try again.")
        
        another = input("\nWould you like to predict another interaction? (y/n): ").lower()
        if another != 'y':
            break
    
    print("\nThank you for using the Genomic Interaction Significance Predictor!")

if __name__ == "__main__":
    main() 