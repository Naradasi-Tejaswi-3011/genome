import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score

def main():
    print("====== GENOMIC INTERACTION PREDICTION - USER TESTING ======")
    
    # Load the original data to extract real values
    try:
        data = pd.read_excel('data/genomic_interactions.xlsx')
        print(f"Loaded original dataset with {len(data)} samples")
        
        # Load the model
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading data or model: {e}")
        return

    # Select 5 random samples from the dataset
    samples = data.sample(n=5, random_state=42)
    
    # Create engineered features for display
    samples['CG_SuppPairs_Ratio'] = samples.apply(
        lambda row: row['CG1_SuppPairs'] / row['CG2_SuppPairs'] if row['CG2_SuppPairs'] > 0 else 0, axis=1)
    samples['CC_SuppPairs_Ratio'] = samples.apply(
        lambda row: row['CC1_SuppPairs'] / row['CC2_SuppPairs'] if row['CC2_SuppPairs'] > 0 else 0, axis=1)
    samples['CN_SuppPairs_Ratio'] = samples.apply(
        lambda row: row['CN1_SuppPairs'] / row['CN2_SuppPairs'] if row['CN2_SuppPairs'] > 0 else 0, axis=1)
    samples['log_distance'] = np.log10(samples['distance'] + 1)
    
    # Define ground truth based on p-value threshold
    P_VALUE_THRESHOLD = 0.005
    samples['is_significant'] = ((samples['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                                (samples['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)

    # Simulating user interaction
    print("\n" + "="*50)
    print("INTERACTIVE TESTING DEMONSTRATION WITH REAL DATA SAMPLES")
    print("="*50)
    
    test_data = []
    
    # Process each sample as if user entered it
    for i, sample in samples.iterrows():
        print(f"\nSAMPLE #{len(test_data) + 1}")
        print(f"Interaction Type: {sample['IntGroup']}")
        print(f"Strand: {sample['Strand']}")
        print(f"Distance: {int(sample['distance'])}")
        print(f"Supporting Pairs (CG1): {int(sample['CG1_SuppPairs'])}")
        print(f"Supporting Pairs (CG2): {int(sample['CG2_SuppPairs'])}")
        print(f"Supporting Pairs (CC1): {int(sample['CC1_SuppPairs'])}")
        print(f"Supporting Pairs (CC2): {int(sample['CC2_SuppPairs'])}")
        print(f"Supporting Pairs (CN1): {int(sample['CN1_SuppPairs'])}")
        print(f"Supporting Pairs (CN2): {int(sample['CN2_SuppPairs'])}")
        print(f"Number of Interactions: {int(sample['NofInts'])}")
        print(f"Annotation: {int(sample['Annotation'])}")
        print(f"Interactor Annotation: {int(sample['InteractorAnnotation'])}")
        
        print("\nProcessing this sample...")
        
        # Prepare entry for prediction
        entry = {
            'IntGroup': sample['IntGroup'],
            'Strand': sample['Strand'],
            'distance': sample['distance'],
            'CG1_SuppPairs': sample['CG1_SuppPairs'],
            'CG2_SuppPairs': sample['CG2_SuppPairs'],
            'CC1_SuppPairs': sample['CC1_SuppPairs'],
            'CC2_SuppPairs': sample['CC2_SuppPairs'],
            'CN1_SuppPairs': sample['CN1_SuppPairs'],
            'CN2_SuppPairs': sample['CN2_SuppPairs'],
            'NofInts': sample['NofInts'],
            'Annotation': sample['Annotation'],
            'InteractorAnnotation': sample['InteractorAnnotation'],
            'Normal': sample['Normal'],
            'CarboplatinTreated': sample['CarboplatinTreated'],
            'GemcitabineTreated': sample['GemcitabineTreated'],
            'CG_SuppPairs_Ratio': sample['CG_SuppPairs_Ratio'],
            'CC_SuppPairs_Ratio': sample['CC_SuppPairs_Ratio'],
            'CN_SuppPairs_Ratio': sample['CN_SuppPairs_Ratio'],
            'log_distance': sample['log_distance']
        }
        
        # Add to test data
        test_data.append(entry)
        
        # Convert to DataFrame for prediction
        entry_df = pd.DataFrame([entry])
        
        # Replace infinite values
        for col in entry_df.columns:
            if entry_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                entry_df[col] = entry_df[col].replace([np.inf, -np.inf], 0)
                entry_df[col] = entry_df[col].fillna(0)
        
        # Make prediction
        try:
            prediction = model.predict(entry_df)[0]
            probability = model.predict_proba(entry_df)[0][1]
            
            print(f"\nPREDICTION RESULT:")
            print(f"Significant Interaction: {'YES' if prediction == 1 else 'NO'}")
            print(f"Probability: {probability:.4f}")
            
            # Compare with ground truth
            actual = sample['is_significant']
            print(f"Actual (based on p-values): {'YES' if actual == 1 else 'NO'}")
            print(f"Prediction {'MATCHES' if prediction == actual else 'DOES NOT MATCH'} actual value")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nWould you like to try another sample? (y/n)")
        print("y [simulated response]")
    
    # Combine all samples into one DataFrame
    all_entries_df = pd.DataFrame(test_data)
    
    # Save all test data
    os.makedirs('results', exist_ok=True)
    all_entries_df.to_excel('results/user_test_samples.xlsx', index=False)
    
    # Final report
    print("\n" + "="*50)
    print("USER TESTING SUMMARY")
    print("="*50)
    print(f"Tested {len(test_data)} samples from the original dataset")
    print(f"Results saved to results/user_test_samples.xlsx")
    print("\nYou can analyze these samples further or test more samples as needed.")

if __name__ == "__main__":
    main() 