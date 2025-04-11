# Genomic Interactions Classification

This project uses Random Forest to classify genomic interactions as significant or non-significant based on supporting pairs, distance, and other features without using p-values during prediction. It includes both a command-line interface and a web application for easy interaction.

## Project Overview

Chromatin interactions in the genome can be classified as significant or non-significant. Traditionally, this classification relies on p-values. This project trains a Random Forest model that can predict the significance of interactions based on other features, making it possible to classify new interactions where p-values might not be available.

## Project Structure

- **data/** - Contains the genomic interactions dataset
- **models/** - Trained model and metadata
- **scripts/** - Python scripts for data analysis, model training, and prediction
- **results/** - Visualization outputs and results
- **docs/** - Documentation files
- **templates/** - HTML templates for the web application
- **app.py** - Flask web application for interactive predictions

## Key Components

### Command Line Tools
1. **scripts/genomic_classification.py** - Main script for data preprocessing, model training, and evaluation
2. **scripts/simple_predict.py** - Predicts significance for interactions in a sample dataset
3. **scripts/user_predict_demo.py** - Interactive demo showing prediction with sample inputs
4. **scripts/dataset_predict.py** - Samples and predicts using existing dataset

### Web Application
The project includes a Flask web application (`app.py`) that provides:
- Modern, responsive web interface
- Interactive form for inputting genomic interaction data
- Sample data loading functionality
- Visual display of prediction results
- Probability visualization

## Dataset

The genomic interactions dataset contains 81,203 interactions with multiple features:
- **Supporting pairs**: CG1_SuppPairs, CG2_SuppPairs, etc.
- **Distances**: Distance between interacting regions
- **P-values**: CG1_p_value, CG2_p_value, etc. (used for determining ground truth)
- **Interaction types**: PP (promoter-promoter), PD (promoter-distal), DD (distal-distal)

## Usage

### Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python scripts/genomic_classification.py
```

### Command Line Predictions

To predict using the sample test data:
```bash
python scripts/simple_predict.py
```

To run the interactive demo:
```bash
python scripts/user_predict_demo.py
```

### Web Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Use the web interface to:
   - Input genomic interaction data
   - Load sample data
   - View prediction results and probabilities

## Model Performance

The Random Forest model achieves approximately 90% accuracy on the test set. The most important features for prediction are:
1. Supporting pair counts
2. Number of interactions
3. Distance between interacting regions
4. Supporting pairs ratio between replicates

## Requirements

- Python 3.6+
- pandas>=1.5.0
- matplotlib>=3.5.0
- seaborn>=0.12.0
- openpyxl>=3.0.0
- numpy>=1.26.0
- scikit-learn>=1.0.0
- imbalanced-learn>=0.10.0
- flask>=2.0.0


