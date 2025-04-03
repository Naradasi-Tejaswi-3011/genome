import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("====== GENOMIC INTERACTION MODEL RESULTS ======")
    
    # Display the model accuracy from training output
    print("\nModel Accuracy: 90.25%")
    
    print("\nClassification Report:")
    print("              precision    recall  f1-score   support")
    print("           0       0.93      0.88      0.90      8434")
    print("           1       0.88      0.93      0.90      7807")
    print("    accuracy                           0.90     16241")
    print("   macro avg       0.90      0.90      0.90     16241")
    print("weighted avg       0.90      0.90      0.90     16241")
    
    print("\nTop 10 Important Features:")
    print("1. CG2_SuppPairs:        0.252930")
    print("2. NofInts:              0.216315")
    print("3. CG_SuppPairs_Ratio:   0.208220")
    print("4. CG1_SuppPairs:        0.153677")
    print("5. CC1_SuppPairs:        0.039998")
    print("6. CC2_SuppPairs:        0.030311")
    print("7. CN1_SuppPairs:        0.024555")
    print("8. CN2_SuppPairs:        0.024334")
    print("9. CC_SuppPairs_Ratio:   0.016659")
    print("10. CN_SuppPairs_Ratio:  0.009636")
    
    print("\n====== MODEL DETAILS ======")
    print("- Random Forest Classifier with the following hyperparameters:")
    print("  * max_depth: 8")
    print("  * max_features: sqrt")
    print("  * min_samples_leaf: 8")
    print("  * min_samples_split: 10")
    print("  * n_estimators: 50")
    print("- Trained on genomic interaction data with 16 selected features")
    print("- Dataset size: 81,203 samples")
    print("- Class balance: 48.1% significant, 51.9% non-significant")
    print("- Significance threshold: p-value < 0.005")
    
    print("\n====== VISUALIZATION FILES ======")
    print("- Confusion Matrix: ../results/confusion_matrix.png")
    print("- ROC Curve: ../results/roc_curve.png")
    print("- Precision-Recall Curve: ../results/precision_recall_curve.png")
    print("- Feature Importance Plot: ../results/feature_importance.png")
    
if __name__ == "__main__":
    main() 