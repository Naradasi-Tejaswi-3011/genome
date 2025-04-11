from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import json
from scripts.genomic_classification import preprocess_data

app = Flask(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
data_dir = os.path.join(current_dir, 'data')
results_dir = os.path.join(current_dir, 'results')


os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


try:
    with open(os.path.join(models_dir, 'random_forest_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, 'model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    print("Model and metadata loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    metadata = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        
        data = {
            'IntGroup': request.form.get('IntGroup'),
            'Strand': request.form.get('Strand'),
            'distance': float(request.form.get('distance')),
            'CG1_SuppPairs': float(request.form.get('CG1_SuppPairs')),
            'CG2_SuppPairs': float(request.form.get('CG2_SuppPairs')),
            'CC1_SuppPairs': float(request.form.get('CC1_SuppPairs')),
            'CC2_SuppPairs': float(request.form.get('CC2_SuppPairs')),
            'CN1_SuppPairs': float(request.form.get('CN1_SuppPairs')),
            'CN2_SuppPairs': float(request.form.get('CN2_SuppPairs')),
            'NofInts': int(request.form.get('NofInts')),
            'Annotation': int(request.form.get('Annotation')),
            'InteractorAnnotation': int(request.form.get('InteractorAnnotation')),
            'CG1_p_value': float(request.form.get('CG1_p_value', 0.001)),
            'CG2_p_value': float(request.form.get('CG2_p_value', 0.001)),
            'CC1_p_value': float(request.form.get('CC1_p_value', 0.001)),
            'CC2_p_value': float(request.form.get('CC2_p_value', 0.001)),
            'CN1_p_value': float(request.form.get('CN1_p_value', 0.001)),
            'CN2_p_value': float(request.form.get('CN2_p_value', 0.001))
        }
        
       
        df = pd.DataFrame([data])
        
       
        df_processed = preprocess_data(df)
        
        
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0][1]
        
       
        result = {
            'prediction': 'YES' if prediction == 1 else 'NO',
            'probability': f"{probability:.4f}"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/sample', methods=['GET'])
def get_sample():
    try:
        
        data_path = os.path.join(data_dir, 'Copy of dataset.xlsx')
        df = pd.read_excel(data_path)
        sample = df.sample(n=1).iloc[0]
        
        
        sample_data = {
            'IntGroup': sample['IntGroup'],
            'Strand': sample['Strand'],
            'distance': int(sample['distance']),
            'CG1_SuppPairs': int(sample['CG1_SuppPairs']),
            'CG2_SuppPairs': int(sample['CG2_SuppPairs']),
            'CC1_SuppPairs': int(sample['CC1_SuppPairs']),
            'CC2_SuppPairs': int(sample['CC2_SuppPairs']),
            'CN1_SuppPairs': int(sample['CN1_SuppPairs']),
            'CN2_SuppPairs': int(sample['CN2_SuppPairs']),
            'NofInts': int(sample['NofInts']),
            'Annotation': int(sample['Annotation']),
            'InteractorAnnotation': int(sample['InteractorAnnotation']),
            'CG1_p_value': float(sample['CG1_p_value']),
            'CG2_p_value': float(sample['CG2_p_value']),
            'CC1_p_value': float(sample['CC1_p_value']),
            'CC2_p_value': float(sample['CC2_p_value']),
            'CN1_p_value': float(sample['CN1_p_value']),
            'CN2_p_value': float(sample['CN2_p_value'])
        }
        
        return jsonify(sample_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 