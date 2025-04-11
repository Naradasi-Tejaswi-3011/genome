import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


P_VALUE_THRESHOLD = 0.005


current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
results_dir = os.path.join(base_dir, 'results')


os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def load_data(file_path=None):
    """Load genomic interaction data from Excel file"""
    if file_path is None:
        file_path = os.path.join(data_dir, 'Copy of dataset.xlsx')
    print(f"Loading data from {file_path}")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the genomic interactions dataset"""
    print("Preprocessing data...")
    

    df_cleaned = df.dropna()
    print(f"Shape after dropping NA values: {df_cleaned.shape}")
    
    
    df_cleaned['is_significant_CG'] = ((df_cleaned['CG1_p_value'] < P_VALUE_THRESHOLD) & 
                                      (df_cleaned['CG2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
    df_cleaned['is_significant_CC'] = ((df_cleaned['CC1_p_value'] < P_VALUE_THRESHOLD) & 
                                      (df_cleaned['CC2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
    df_cleaned['is_significant_CN'] = ((df_cleaned['CN1_p_value'] < P_VALUE_THRESHOLD) & 
                                      (df_cleaned['CN2_p_value'] < P_VALUE_THRESHOLD)).astype(int)
    

    df_cleaned['is_significant'] = df_cleaned['is_significant_CG']
    
    df_cleaned['CG_SuppPairs_Ratio'] = np.where(
        df_cleaned['CG2_SuppPairs'] > 0,
        np.minimum(df_cleaned['CG1_SuppPairs'] / df_cleaned['CG2_SuppPairs'], 10),
        0
    )
    
    df_cleaned['CC_SuppPairs_Ratio'] = np.where(
        df_cleaned['CC2_SuppPairs'] > 0,
        np.minimum(df_cleaned['CC1_SuppPairs'] / df_cleaned['CC2_SuppPairs'], 10),
        0
    )
    
    df_cleaned['CN_SuppPairs_Ratio'] = np.where(
        df_cleaned['CN2_SuppPairs'] > 0,
        np.minimum(df_cleaned['CN1_SuppPairs'] / df_cleaned['CN2_SuppPairs'], 10),
        0
    )
    
    df_cleaned['log_distance'] = np.log1p(df_cleaned['distance'])
    
    return df_cleaned

def feature_selection(df):
    """Select and engineer features for the model"""
    print("Selecting features...")
    
   
    categorical_cols = ['IntGroup', 'Strand']
    
   
    exclude_cols = [
        'is_significant', 'is_significant_CG', 'is_significant_CC', 'is_significant_CN',
        'Feature_Chr', 'Feature_Start', 'RefSeqName', 'TranscriptName', 
        'InteractorName', 'InteractorID', 'Interactor_Chr', 'Interactor_Start', 'Interactor_End',
        'GemcitabineTreated', 'CarboplatinTreated', 'Normal'  
    ]
    

    p_value_cols = [col for col in df.columns if 'p_value' in col]
    
    
    numerical_cols = [col for col in df.columns if col not in exclude_cols + categorical_cols + p_value_cols 
                      and df[col].dtype in ['int64', 'float64']]
    
    # Add noise to supporting pairs
    noise_level = 0.3  
    for col in ['CG1_SuppPairs', 'CG2_SuppPairs', 'CC1_SuppPairs', 'CC2_SuppPairs', 
                'CN1_SuppPairs', 'CN2_SuppPairs']:
        noise = np.random.normal(0, noise_level, len(df))
        df[col] = df[col] * (1 + noise)
        df[col] = np.maximum(df[col], 0)  
    
    # Clean up any infinite or NaN values
    for col in numerical_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].fillna(0)
    
    print(f"Using categorical features: {categorical_cols}")
    print(f"Using numerical features (excluding p-values): {numerical_cols}")
    print(f"P-value features (for training only): {p_value_cols}")
    
   
    X = df[numerical_cols + categorical_cols]
    y = df['is_significant']
    
    print(f"Selected {len(numerical_cols) + len(categorical_cols)} features for prediction")
    print(f"Class distribution: {y.value_counts()}")
    
    return X, y, numerical_cols, categorical_cols, p_value_cols

def balance_dataset(X, y):
    """Balance the dataset using SMOTE if it's imbalanced"""
    print("Checking class balance...")
    class_counts = y.value_counts()
    
    if min(class_counts) / sum(class_counts) < 0.4:
        print("Dataset is imbalanced. Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"Original class distribution: {pd.Series(y).value_counts()}")
        print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts()}")
        return X_balanced, y_balanced
    
    print("Dataset is sufficiently balanced. No resampling needed.")
    return X, y

def train_random_forest(X, y, numerical_cols, categorical_cols, p_value_cols, df):
    """Train a Random Forest model with preprocessing pipeline"""
    print("Training Random Forest model with preprocessing pipeline...")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50],
        'classifier__max_depth': [8],
        'classifier__min_samples_split': [10],
        'classifier__min_samples_leaf': [8],
        'classifier__max_features': ['sqrt']
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    model = grid_search.best_estimator_
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Get feature importance
    feature_names = numerical_cols.copy()
    encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    for i, (name, cats) in enumerate(zip(categorical_cols, encoder.categories_)):
        feature_names.extend([f"{name}_{val}" for val in cats[1:]])
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.named_steps['classifier'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()
    
    print("\nTop 10 important features:")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\nSaving model and metadata...")
    # Save the model
    with open(os.path.join(models_dir, 'random_forest_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as '{os.path.join(models_dir, 'random_forest_model.pkl')}'")

    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'categorical_features': categorical_cols,
        'numerical_features': numerical_cols,
        'p_value_features': p_value_cols,
        'feature_importance': feature_importance.to_dict(),
        'best_params': model.get_params(),
        'accuracy': accuracy,
        'classification_report': classification_report
    }
    
    metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Model metadata saved as '{metadata_path}'")
    
    print("\nAnalysis complete! All plots saved in the results folder:")
    print(f"1. Confusion Matrix: {os.path.join(results_dir, 'confusion_matrix.png')}")
    print(f"2. ROC Curve: {os.path.join(results_dir, 'roc_curve.png')}")
    print(f"3. Precision-Recall Curve: {os.path.join(results_dir, 'precision_recall_curve.png')}")
    print(f"4. Feature Importance: {os.path.join(results_dir, 'feature_importance.png')}")
    
    return model, feature_importance

def main():
    """Main function to run the genomic classification pipeline"""
    print("Starting genomic classification pipeline...")
    
    # Create necessary directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Feature selection
    X, y, numerical_cols, categorical_cols, p_value_cols = feature_selection(df_processed)
    
    # Balance dataset
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Train model
    model, feature_importance = train_random_forest(X_balanced, y_balanced, numerical_cols, categorical_cols, p_value_cols, df_processed)
    
    print("\nAnalysis complete! All plots saved in the results folder:")
    print(f"1. Confusion Matrix: {os.path.join(results_dir, 'confusion_matrix.png')}")
    print(f"2. ROC Curve: {os.path.join(results_dir, 'roc_curve.png')}")
    print(f"3. Precision-Recall Curve: {os.path.join(results_dir, 'precision_recall_curve.png')}")
    print(f"4. Feature Importance: {os.path.join(results_dir, 'feature_importance.png')}")

def predict_interactions(file_path):
    """
    Predict interactions for new data using the trained model.
    
    Args:
        file_path (str): Path to the Excel file containing new data
    """
    try:
        model_path = os.path.join(models_dir, 'random_forest_model.pkl')
        metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        new_data = pd.read_excel(file_path)
        X_new = preprocess_data(new_data)
        
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        
        new_data['Predicted_Interaction'] = predictions
        new_data['Interaction_Probability'] = probabilities[:, 1]
        
        output_path = os.path.join(results_dir, 'predictions.xlsx')
        new_data.to_excel(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        print("\nPrediction Summary:")
        print(f"Total samples: {len(new_data)}")
        print(f"Predicted interactions: {sum(predictions)}")
        print(f"Predicted non-interactions: {len(predictions) - sum(predictions)}")
        
        return new_data
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    main() 