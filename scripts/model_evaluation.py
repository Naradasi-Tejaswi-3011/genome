import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data')
models_dir = os.path.join(project_root, 'models')
results_dir = os.path.join(project_root, 'results')


print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")
print(f"Models directory: {models_dir}")
print(f"Results directory: {results_dir}")

def main():
    print("====== GENOMIC INTERACTION MODEL EVALUATION ON TEST SET ======")
    
    try:
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        
        dataset_path = os.path.join(data_dir, 'Copy of dataset.xlsx')
        print(f"Attempting to load dataset from: {dataset_path}")
        data = pd.read_excel(dataset_path)
        print(f"Loaded original dataset with {len(data)} samples")
        
        
        model_path = os.path.join(models_dir, 'random_forest_model.pkl')
        print(f"Attempting to load model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading data or model: {e}")
        return

    
    P_VALUE_THRESHOLD = 0.005
    
  
    data = data.dropna()
    
    
    data['is_significant'] = ((data['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                             (data['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
    
  
    data['CG_SuppPairs_Ratio'] = data.apply(
        lambda row: row['CG1_SuppPairs'] / row['CG2_SuppPairs'] if row['CG2_SuppPairs'] > 0 else 0, axis=1)
    data['CC_SuppPairs_Ratio'] = data.apply(
        lambda row: row['CC1_SuppPairs'] / row['CC2_SuppPairs'] if row['CC2_SuppPairs'] > 0 else 0, axis=1)
    data['CN_SuppPairs_Ratio'] = data.apply(
        lambda row: row['CN1_SuppPairs'] / row['CN2_SuppPairs'] if row['CN2_SuppPairs'] > 0 else 0, axis=1)
    data['log_distance'] = np.log10(data['distance'] + 1)
    
    
    for col in data.columns:
        if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            data[col] = data[col].replace([np.inf, -np.inf], 0)
            data[col] = data[col].fillna(0)
    

    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Test set size: {len(test_data)} samples")
    

    os.makedirs(results_dir, exist_ok=True)
    test_data.to_excel(os.path.join(results_dir, 'test_set.xlsx'), index=False)
    
   
    print("\nMaking predictions on test set...")
    try:
        
        y_true = test_data['is_significant']
        
        categorical_cols = ['IntGroup', 'Strand']
        
        
        exclude_cols = [
            'is_significant', 'is_significant_CG', 'is_significant_CC', 'is_significant_CN',
            'Feature_Chr', 'Feature_Start', 'RefSeqName', 'TranscriptName', 
            'InteractorName', 'InteractorID', 'Interactor_Chr', 'Interactor_Start', 'Interactor_End',
            'GemcitabineTreated', 'CarboplatinTreated', 'Normal'
        ]
        
      
        p_value_cols = [col for col in test_data.columns if 'p_value' in col]
        
       
        numerical_cols = [col for col in test_data.columns if col not in exclude_cols + categorical_cols + p_value_cols 
                         and test_data[col].dtype in ['int64', 'float64']]
        
        engineered_features = ['CG_SuppPairs_Ratio', 'CC_SuppPairs_Ratio', 'CN_SuppPairs_Ratio', 'log_distance']
        for feature in engineered_features:
            if feature not in numerical_cols:
                numerical_cols.append(feature)
        
        X_test = test_data[numerical_cols + categorical_cols]
        
        for col in X_test.columns:
            if X_test[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X_test[col] = X_test[col].replace([np.inf, -np.inf], 0)
                X_test[col] = X_test[col].fillna(0)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy on test set: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        test_data['predicted_class'] = y_pred
        test_data['prediction_probability'] = y_prob
        
        test_data.to_excel(os.path.join(results_dir, 'test_results.xlsx'), index=False)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(results_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        
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
        plt.savefig(os.path.join(results_dir, 'test_roc_curve.png'), dpi=300, bbox_inches='tight')
        
        
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
           
            preprocessor = model.named_steps['preprocessor']
            feature_names = []
            
            
            feature_names.extend(numerical_cols)
            
      
            if hasattr(preprocessor, 'transformers_'):
                for name, transformer, cols in preprocessor.transformers_:
                    if name == 'cat':
                        if hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
                            encoder = transformer.named_steps['encoder']
                            if hasattr(encoder, 'get_feature_names_out'):
                                cat_features = encoder.get_feature_names_out(categorical_cols)
                                feature_names.extend(cat_features)
            
           
            importances = model.named_steps['classifier'].feature_importances_
            
          
            indices = np.argsort(importances)[-20:]  
            
          
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'test_feature_importance.png'), dpi=300, bbox_inches='tight')
            
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