import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os

def main():
    print("====== GENOMIC INTERACTION MODEL EVALUATION ON TEST SET ======")
    
    # Load the original data
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

    # Preprocess data the same way as in training
    P_VALUE_THRESHOLD = 0.005
    
    # Clean data
    data = data.dropna()
    
    # Create target variable based on p-value threshold
    data['is_significant'] = ((data['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                             (data['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
    
    # Feature engineering
    data['CG_SuppPairs_Ratio'] = data.apply(
        lambda row: row['CG1_SuppPairs'] / row['CG2_SuppPairs'] if row['CG2_SuppPairs'] > 0 else 0, axis=1)
    data['CC_SuppPairs_Ratio'] = data.apply(
        lambda row: row['CC1_SuppPairs'] / row['CC2_SuppPairs'] if row['CC2_SuppPairs'] > 0 else 0, axis=1)
    data['CN_SuppPairs_Ratio'] = data.apply(
        lambda row: row['CN1_SuppPairs'] / row['CN2_SuppPairs'] if row['CN2_SuppPairs'] > 0 else 0, axis=1)
    data['log_distance'] = np.log10(data['distance'] + 1)
    
    # Handle infinite values
    for col in data.columns:
        if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            data[col] = data[col].replace([np.inf, -np.inf], 0)
            data[col] = data[col].fillna(0)
    
    # Create test set (20% of data)
    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Test set size: {len(test_data)} samples")
    
    # Save test data
    os.makedirs('results', exist_ok=True)
    test_data.to_excel('results/test_set.xlsx', index=False)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    try:
        # Get X and y from test data
        y_true = test_data['is_significant']
        
        # Select features
        categorical_cols = ['IntGroup', 'Strand']
        
        # Exclude features that aren't used for prediction
        exclude_cols = [
            'is_significant', 'is_significant_CG', 'is_significant_CC', 'is_significant_CN',
            'Feature_Chr', 'Feature_Start', 'RefSeqName', 'TranscriptName', 
            'InteractorName', 'InteractorID', 'Interactor_Chr', 'Interactor_Start', 'Interactor_End',
            'GemcitabineTreated', 'CarboplatinTreated', 'Normal'
        ]
        
        # Remove p-value columns
        p_value_cols = [col for col in test_data.columns if 'p_value' in col]
        
        # Get numerical columns
        numerical_cols = [col for col in test_data.columns if col not in exclude_cols + categorical_cols + p_value_cols 
                         and test_data[col].dtype in ['int64', 'float64']]
        
        # Add engineered features
        engineered_features = ['CG_SuppPairs_Ratio', 'CC_SuppPairs_Ratio', 'CN_SuppPairs_Ratio', 'log_distance']
        for feature in engineered_features:
            if feature not in numerical_cols:
                numerical_cols.append(feature)
        
        # Get X
        X_test = test_data[numerical_cols + categorical_cols]
        
        # Handle infinite values in X_test
        for col in X_test.columns:
            if X_test[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X_test[col] = X_test[col].replace([np.inf, -np.inf], 0)
                X_test[col] = X_test[col].fillna(0)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy on test set: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Add predictions to test data
        test_data['predicted_class'] = y_pred
        test_data['prediction_probability'] = y_prob
        
        # Save results
        test_data.to_excel('results/test_results.xlsx', index=False)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('results/test_roc_curve.png', dpi=300, bbox_inches='tight')
        
        # Print feature importance
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            # Get feature names
            preprocessor = model.named_steps['preprocessor']
            feature_names = []
            
            # Get numerical feature names
            feature_names.extend(numerical_cols)
            
            # Get one-hot encoded feature names
            if hasattr(preprocessor, 'transformers_'):
                for name, transformer, cols in preprocessor.transformers_:
                    if name == 'cat':
                        if hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
                            encoder = transformer.named_steps['encoder']
                            if hasattr(encoder, 'get_feature_names_out'):
                                cat_features = encoder.get_feature_names_out(categorical_cols)
                                feature_names.extend(cat_features)
            
            # Get feature importances
            importances = model.named_steps['classifier'].feature_importances_
            
            # Sort feature importances
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            # Plot feature importances
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig('results/test_feature_importance.png', dpi=300, bbox_inches='tight')
            
            print("\nTop 5 feature importances:")
            for i in indices[-5:]:
                feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
                print(f"{feature_name}: {importances[i]:.4f}")
                
        print("\nEvaluation complete! Results saved to results/test_results.xlsx")
        print("Visualizations saved to results/ directory")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 