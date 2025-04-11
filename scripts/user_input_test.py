import pandas as pd
import numpy as np
import pickle
import os

def get_input(prompt, default=None, input_type=str):
    """Get user input with a default value"""
    if default is not None:
        user_input = input(f"{prompt} [{default}]: ")
        if user_input.strip() == "":
            return default
    else:
        user_input = input(f"{prompt}: ")
    

    if input_type == int:
        return int(user_input) if user_input.strip() else default
    elif input_type == float:
        return float(user_input) if user_input.strip() else default
    else:
        return user_input if user_input.strip() else default
def main():
    print("====== GENOMIC INTERACTION PREDICTION - INTERACTIVE USER TESTING ======")
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")

       
        try:
            data = pd.read_excel('data/genomic_interactions.xlsx', nrows=5)
            print("Loaded sample data for reference")
            print("\nSample values from dataset (for reference):")
            print(f"IntGroup examples: {data['IntGroup'].unique().tolist()}")
            print(f"Strand examples: {data['Strand'].unique().tolist()}")
            print(f"Distance range: {data['distance'].min()} to {data['distance'].max()}")
            print(f"Supporting Pairs range: {data['CG1_SuppPairs'].min()} to {data['CG1_SuppPairs'].max()}")
        except Exception as e:
            print(f"Note: Could not load reference data: {e}")
            print("You will need to provide all values manually.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    continue_testing = True
    test_data = []
    
    print("\n" + "="*50)
    print("INTERACTIVE TESTING")
    print("="*50)
    print("Enter genomic interaction parameters to test the model.")
    print("Press Enter to use example values in brackets, or enter your own.")
    
    while continue_testing:
        print("\n--- NEW PREDICTION ---")
        
       
        int_group = get_input("Enter Interaction Type (PP, PD, or DD)", "PP")
        strand = get_input("Enter Strand (+ or -)", "+")
        distance = get_input("Enter Distance (typical range: 1000-1000000)", 50000, int)
        
        cg1_pairs = get_input("Enter Supporting Pairs CG1 (typical range: 0-150)", 50, int)
        cg2_pairs = get_input("Enter Supporting Pairs CG2 (typical range: 0-150)", 40, int)
        cc1_pairs = get_input("Enter Supporting Pairs CC1 (typical range: 0-150)", 60, int)
        cc2_pairs = get_input("Enter Supporting Pairs CC2 (typical range: 0-150)", 55, int)
        cn1_pairs = get_input("Enter Supporting Pairs CN1 (typical range: 0-150)", 45, int)
        cn2_pairs = get_input("Enter Supporting Pairs CN2 (typical range: 0-150)", 40, int)
        
        nof_ints = get_input("Enter Number of Interactions (typical range: 1-10)", 2, int)
        annotation = get_input("Enter Annotation (0-3)", 1, int)
        interactor_annotation = get_input("Enter Interactor Annotation (0-3)", 2, int)
        
        normal = get_input("Is it Normal tissue? (0=no, 1=yes)", 0, int)
        carboplatin = get_input("Is it Carboplatin Treated? (0=no, 1=yes)", 0, int)
        gemcitabine = get_input("Is it Gemcitabine Treated? (0=no, 1=yes)", 1, int)
        
        cg_ratio = cg1_pairs / cg2_pairs if cg2_pairs > 0 else 0
        cc_ratio = cc1_pairs / cc2_pairs if cc2_pairs > 0 else 0
        cn_ratio = cn1_pairs / cn2_pairs if cn2_pairs > 0 else 0
        log_distance = np.log10(distance + 1)
        
        print("\nEngineered Features:")
        print(f"CG Supporting Pairs Ratio: {cg_ratio:.4f}")
        print(f"CC Supporting Pairs Ratio: {cc_ratio:.4f}")
        print(f"CN Supporting Pairs Ratio: {cn_ratio:.4f}")
        print(f"Log Distance: {log_distance:.4f}")
        
      
        entry = {
            'IntGroup': int_group,
            'Strand': strand,
            'distance': distance,
            'CG1_SuppPairs': cg1_pairs,
            'CG2_SuppPairs': cg2_pairs,
            'CC1_SuppPairs': cc1_pairs,
            'CC2_SuppPairs': cc2_pairs,
            'CN1_SuppPairs': cn1_pairs,
            'CN2_SuppPairs': cn2_pairs,
            'NofInts': nof_ints,
            'Annotation': annotation,
            'InteractorAnnotation': interactor_annotation,
            'Normal': normal,
            'CarboplatinTreated': carboplatin,
            'GemcitabineTreated': gemcitabine,
            'CG_SuppPairs_Ratio': cg_ratio,
            'CC_SuppPairs_Ratio': cc_ratio,
            'CN_SuppPairs_Ratio': cn_ratio,
            'log_distance': log_distance
        }
        
        test_data.append(entry)
        
        entry_df = pd.DataFrame([entry])
        
        for col in entry_df.columns:
            if entry_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                entry_df[col] = entry_df[col].replace([np.inf, -np.inf], 0)
                entry_df[col] = entry_df[col].fillna(0)
        
        print("\nProcessing input data...")
        
        try:
            prediction = model.predict(entry_df)[0]
            probability = model.predict_proba(entry_df)[0][1]
            
            print("\n" + "="*30)
            print("PREDICTION RESULT")
            print("="*30)
            print(f"Significant Interaction: {'YES' if prediction == 1 else 'NO'}")
            print(f"Probability: {probability:.4f}")
            
            if probability > 0.75:
                print("Confidence: High")
            elif probability > 0.6:
                print("Confidence: Medium")
            else:
                print("Confidence: Low")
                
            entry['prediction'] = prediction
            entry['probability'] = probability
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
        
        
        continue_input = input("\nWould you like to test another set of values? (y/n): ")
        continue_testing = continue_input.lower() == 'y'
    
    if test_data:
        all_entries_df = pd.DataFrame(test_data)
        
        os.makedirs('results', exist_ok=True)
        filename = 'results/user_input_test_results.xlsx'
        all_entries_df.to_excel(filename, index=False)
        print(f"\nAll test results have been saved to {filename}")
    
    print("\nThank you for using the Genomic Interaction Prediction Tool!")

if __name__ == "__main__":
    main() 