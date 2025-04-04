# Genomic Interactions Classification

This project uses Random Forest to classify genomic interactions as significant or non-significant based on supporting pairs, distance, and other features without using p-values during prediction.

## Project Overview

Chromatin interactions in the genome can be classified as significant or non-significant. Traditionally, this classification relies on p-values. This project trains a Random Forest model that can predict the significance of interactions based on other features, making it possible to classify new interactions where p-values might not be available.

## Project Structure

- **data/** - Contains the genomic interactions dataset
- **models/** - Trained model and metadata
- **scripts/** - Python scripts for data analysis, model training, and prediction
- **results/** - Visualization outputs and results
- **docs/** - Documentation files

## Key Scripts

1. **scripts/genomic_classification.py** - Main script for data preprocessing, model training, and evaluation
2. **scripts/simple_predict.py** - Predicts significance for interactions in a sample dataset
3. **scripts/user_predict_demo.py** - Interactive demo showing prediction with sample inputs
4. **scripts/dataset_predict.py** - Samples and predicts using existing dataset

## Dataset

The genomic interactions dataset contains 81,203 interactions with multiple features:
- **Supporting pairs**: CG1_SuppPairs, CG2_SuppPairs, etc.
- **Distances**: Distance between interacting regions
- **P-values**: CG1_p_value, CG2_p_value, etc. (used for determining ground truth)
- **Interaction types**: PP (promoter-promoter), PD (promoter-distal), DD (distal-distal)

## Usage

### Training the Model

```bash
python scripts/genomic_classification.py
```

### Making Predictions

To predict using the sample test data:
```bash
python scripts/simple_predict.py
```

To run the interactive demo:
```bash
python scripts/user_predict_demo.py
```

## Model Performance

The Random Forest model achieves approximately 90% accuracy on the test set. The most important features for prediction are:
1. Supporting pair counts
2. Number of interactions
3. Distance between interacting regions
4. Supporting pairs ratio between replicates

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn

