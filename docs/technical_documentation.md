# Technical Documentation: Genomic Interactions Classification

## 1. Introduction

This document provides technical details about the genomic interactions classification project, which uses machine learning to identify significant chromatin interactions without relying on p-values during prediction.

## 2. Data Description

The dataset contains 81,203 chromatin interactions from Hi-C experiments with the following key features:

| Feature Type | Description | Examples |
|--------------|-------------|----------|
| Genomic Locations | Chromosome, start, end positions | Feature_Chr, Feature_Start, Interactor_Chr, Interactor_Start |
| Supporting Pairs | Read pairs supporting the interaction | CG1_SuppPairs, CG2_SuppPairs, CC1_SuppPairs, etc. |
| P-values | Statistical significance | CG1_p_value, CG2_p_value, etc. |
| Annotation | Genomic annotation of interactions | IntGroup (PP/PD/DD) |
| Distance | Genomic distance between interacting regions | distance |
| Treatment | Presence in different treatment conditions | Normal, CarboplatinTreated, GemcitabineTreated |

### 2.1 Target Definition

An interaction is considered significant if p-values in both replicates are below the threshold of 0.005:
```
is_significant = (CG1_p_value < 0.005) & (CG2_p_value < 0.005)
```

## 3. Methodology

### 3.1 Data Preprocessing

1. **Feature Engineering**:
   - Supporting pair ratios between replicates
   - Log-transformed distance
   - One-hot encoding of categorical variables

2. **Noise Addition**: 
   - Small amount of noise (30%) added to supporting pairs to prevent overfitting

3. **Feature Selection**:
   - Training: All features except genomic coordinates
   - Prediction: All features except p-values and treatment-specific features

### 3.2 Model Architecture

The project uses a Random Forest classifier with the following hyperparameters:

- n_estimators: 50
- max_depth: 8
- min_samples_split: 10
- min_samples_leaf: 8
- max_features: sqrt
- class_weight: balanced

### 3.3 Model Training

- Train-test split: 80% training, 20% testing
- Stratified sampling to maintain class distribution
- 5-fold cross-validation for hyperparameter tuning
- Preprocessing pipeline with scaling for numerical features and one-hot encoding for categorical features

## 4. Model Evaluation

### 4.1 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 90.4% |
| Precision (class 0) | 0.93 |
| Recall (class 0) | 0.88 |
| Precision (class 1) | 0.88 |
| Recall (class 1) | 0.93 |
| F1-score | 0.90 |

### 4.2 Feature Importance

The top 5 most important features:
1. CG2_SuppPairs (0.25)
2. NofInts (0.21)
3. CG_SuppPairs_Ratio (0.20)
4. CG1_SuppPairs (0.17)
5. CC1_SuppPairs (0.04)

## 5. Implementation Details

### 5.1 Prediction Pipeline

The prediction process consists of:
1. Feature preparation (including engineered features)
2. Data normalization
3. Model inference
4. Probability estimation

### 5.2 File Structure

- **genomic_classification.py**: Main script for training
- **simple_predict.py**: Batch prediction on sample data
- **user_predict_demo.py**: Interactive prediction with sample inputs
- **test_model.py**: Tests model on held-out data

## 6. Limitations and Future Work

### 6.1 Limitations

- Current model only focuses on Gemcitabine treatment conditions
- Model doesn't incorporate sequence-specific features
- Limited by the experimental design of the input data

### 6.2 Future Improvements

- Include additional epigenomic features (ChIP-seq data)
- Implement deep learning approaches (e.g., graph neural networks)
- Integrate with biological pathway analysis
- Expand to multi-condition analysis 