"""
Flask Web Application for Alzheimer's Disease Prediction
Provides a web interface for making predictions using the trained ML model
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import AlzheimerPredictor
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)

# Global predictor instance (loaded once at startup)
predictor = None


def load_model():
    """Load the trained model at startup"""
    global predictor
    try:
        # Try to load tuned model first, fallback to baseline
        model_paths = [
            os.path.join('..', 'models', 'tuned_models', 'random_forest_tuned.pkl'),
            os.path.join('..', 'models', 'tuned_models', 'gradient_boosting_tuned.pkl'),
            os.path.join('..', 'models', 'saved_models', 'gradient_boosting_model.pkl'),
            os.path.join('..', 'models', 'saved_models', 'random_forest_model.pkl')
        ]
        
        scaler_path = os.path.join('..', 'data', 'processed', 'scaler.pkl')
        feature_names_path = os.path.join('..', 'data', 'processed', 'feature_names.pkl')
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                predictor = AlzheimerPredictor()
                predictor.load_model(model_path)
                
                # Load scaler and feature names if available
                if os.path.exists(scaler_path):
                    predictor.load_scaler(scaler_path)
                if os.path.exists(feature_names_path):
                    predictor.load_feature_names(feature_names_path)
                
                print(f"[SUCCESS] Model loaded from: {model_path}")
                return True
        
        print("[ERROR] No model found!")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        traceback.print_exc()
        return False


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Expects JSON data with patient features
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Validate required fields
        required_fields = [
            'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
            'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
            'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
            'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
            'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
            'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems',
            'ADL', 'Confusion', 'Disorientation', 'PersonalityChanges',
            'DifficultyCompletingTasks', 'Forgetfulness'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'success': False
            }), 400
        
        # Make prediction
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        result = predictor.predict_with_details(data)
        
        # Format response
        response = {
            'success': True,
            'prediction': {
                'diagnosis': result['diagnosis'],
                'diagnosis_label': 'Alzheimer\'s Disease' if result['diagnosis'] == 1 else 'No Alzheimer\'s',
                'confidence': float(result['confidence']),
                'probability_no_alzheimers': float(result['probabilities'][0]),
                'probability_alzheimers': float(result['probabilities'][1])
            },
            'risk_level': get_risk_level(result['confidence'], result['diagnosis']),
            'recommendations': get_recommendations(result['diagnosis'], data)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500


def get_risk_level(confidence, diagnosis):
    """Determine risk level based on confidence and diagnosis"""
    if diagnosis == 0:
        return 'Low Risk'
    else:
        if confidence >= 0.9:
            return 'High Risk'
        elif confidence >= 0.7:
            return 'Moderate Risk'
        else:
            return 'Low-Moderate Risk'


def get_recommendations(diagnosis, patient_data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if diagnosis == 1:
        recommendations.append("Consult with a neurologist for comprehensive evaluation")
        recommendations.append("Consider cognitive assessment and brain imaging")
        recommendations.append("Discuss treatment options with healthcare provider")
    else:
        recommendations.append("Maintain regular health checkups")
        recommendations.append("Continue healthy lifestyle habits")
    
    # Lifestyle recommendations
    if patient_data.get('Smoking', 0) == 1:
        recommendations.append("Consider smoking cessation programs")
    
    if patient_data.get('PhysicalActivity', 0) < 5:
        recommendations.append("Increase physical activity to at least 150 minutes per week")
    
    if patient_data.get('SleepQuality', 0) < 6:
        recommendations.append("Improve sleep quality through better sleep hygiene")
    
    if patient_data.get('DietQuality', 0) < 6:
        recommendations.append("Adopt a Mediterranean or MIND diet for brain health")
    
    # Cognitive health
    if patient_data.get('MMSE', 30) < 24:
        recommendations.append("Engage in cognitive stimulation activities")
        recommendations.append("Consider memory training exercises")
    
    return recommendations


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    }), 200


if __name__ == '__main__':
    print("="*80)
    print("ALZHEIMER'S DISEASE PREDICTION WEB APPLICATION")
    print("="*80)
    
    # Load model at startup
    if load_model():
        print("\n[INFO] Starting Flask server...")
        print("[INFO] Access the application at: http://localhost:5000")
        print("="*80)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n[ERROR] Failed to start application - model could not be loaded")
        print("[INFO] Please ensure the model is trained and saved")
