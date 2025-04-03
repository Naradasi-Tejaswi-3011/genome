import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from test_model import predict_significance

def load_model_and_metadata():
    """Load the trained model and necessary metadata"""
    # Load the trained model
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Loaded saved model")
    except FileNotFoundError:
        print("Error: Model file 'random_forest_model.pkl' not found.")
        print("Please run 'python genomic_classification.py' first to train and save the model.")
        return None, None, None
    
    # Load feature information
    try:
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            numerical_cols = metadata['numerical_cols']
            categorical_cols = metadata['categorical_cols']
        print("Loaded model metadata")
    except FileNotFoundError:
        # If metadata file doesn't exist, use default feature lists
        numerical_cols = [
            'Annotation', 'InteractorAnnotation', 'distance',
            'CG1_SuppPairs', 'CG2_SuppPairs', 'CC1_SuppPairs', 'CC2_SuppPairs',
            'CN1_SuppPairs', 'CN2_SuppPairs', 'NofInts'
        ]
        categorical_cols = ['IntGroup', 'Strand']
        print("Using default feature lists")
    
    return model, numerical_cols, categorical_cols

def predict_from_file(input_file, model, numerical_cols, categorical_cols):
    """Predict significance for interactions in the input file"""
    # Load input data
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        input_data = pd.read_excel(input_file)
    else:
        input_data = pd.read_csv(input_file)
    
    print(f"Loaded input data with {len(input_data)} rows")
    
    # Make predictions
    results = predict_significance(model, input_data, numerical_cols, categorical_cols)
    
    # Save predictions to file
    output_file = input_file.replace('.xlsx', '_predictions.xlsx').replace('.csv', '_predictions.csv')
    if output_file.endswith('.xlsx'):
        results.to_excel(output_file, index=False)
    else:
        results.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    
    return results

def main():
    print("Genomic Interactions Significance Predictor")
    print("===========================================")
    
    # Load model and metadata
    model, numerical_cols, categorical_cols = load_model_and_metadata()
    if model is None:
        return
    
    # Get input file from user
    while True:
        try:
            input_file = input("\nEnter the path to your input file (Excel or CSV): ")
            if not input_file:
                print("Operation canceled. Exiting...")
                return
            
            # Make predictions
            results = predict_from_file(input_file, model, numerical_cols, categorical_cols)
            
            # Summarize results
            sig_count = results['predicted_significant'].sum()
            total = len(results)
            print(f"\nSummary: {sig_count} significant interactions out of {total} ({sig_count/total:.1%})")
            
            # Plot distribution of prediction probabilities
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(results['probability_significant'], bins=20, alpha=0.7)
            plt.axvline(0.5, color='red', linestyle='--')
            plt.xlabel('Probability of Significance')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Probabilities')
            
            plt.subplot(1, 2, 2)
            plt.scatter(results['CG1_SuppPairs'], results['CG2_SuppPairs'], 
                      c=results['probability_significant'], cmap='viridis', 
                      alpha=0.7, s=50)
            plt.colorbar(label='Probability')
            plt.xlabel('CG1 Supporting Pairs')
            plt.ylabel('CG2 Supporting Pairs')
            plt.title('Prediction Probabilities by Supporting Pairs')
            
            plt.tight_layout()
            viz_file = input_file.replace('.xlsx', '_viz.png').replace('.csv', '_viz.png')
            plt.savefig(viz_file)
            print(f"Visualization saved to {viz_file}")
            
            # Ask if user wants to predict more files
            another = input("\nPredict another file? (y/n): ").lower()
            if another != 'y':
                break
                
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please check your input file and try again.")
    
    print("\nThank you for using the Genomic Interactions Significance Predictor!")

if __name__ == "__main__":
    main() 