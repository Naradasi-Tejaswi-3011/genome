# Genomic Interaction Predictor Web Application

This web application provides a user-friendly interface for the Genomic Interaction Predictor. It allows users to input genomic interaction data and get predictions about whether the interactions are significant, without having to use the command line.

## Features

- Modern, responsive web interface
- Input form for all required parameters
- "Load Sample" button to populate the form with a random sample from the dataset
- Visual display of prediction results
- Probability visualization
- Comparison with actual significance based on p-values

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Train the model first by running:
   ```
   python scripts/genomic_classification.py
   ```

## Running the Web Application

1. Navigate to the project directory
2. Run the Flask application:
   ```
   python app.py
   ```
3. Open your web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Using the Web Interface

1. Fill in the form with your genomic interaction data
2. Click the "Predict" button to get the prediction
3. Alternatively, click "Load Sample" to populate the form with a random sample from the dataset
4. View the prediction results in the right panel

## How It Works

The web application uses the same Random Forest model trained by the `genomic_classification.py` script. It provides a graphical interface for:

1. Inputting the same parameters that would be used in the command-line version
2. Displaying the prediction results in a more user-friendly format
3. Showing the probability of the prediction
4. Comparing the prediction with the actual significance based on p-values

## Troubleshooting

- If you see an error about the model not being loaded, make sure you've run `genomic_classification.py` first
- If the application doesn't start, check that Flask is installed correctly
- For any other issues, check the console output for error messages
