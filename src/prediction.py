"""
Prediction Module
Handles making predictions on new data using trained models
"""

import pandas as pd
import numpy as np
import joblib
import os


class AlzheimerPredictor:
    """
    Makes predictions for Alzheimer's disease using trained models
    """
    
    def __init__(self, model_path=None, scaler_path=None, feature_names_path=None):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
            feature_names_path (str): Path to saved feature names
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
        if feature_names_path:
            self.load_feature_names(feature_names_path)
    
    def load_model(self, model_path):
        """
        Load trained model
        
        Args:
            model_path (str): Path to the model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def load_scaler(self, scaler_path):
        """
        Load fitted scaler
        
        Args:
            scaler_path (str): Path to the scaler file
        """
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    
    def load_feature_names(self, feature_names_path):
        """
        Load feature names
        
        Args:
            feature_names_path (str): Path to feature names file
        """
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
        
        self.feature_names = joblib.load(feature_names_path)
        print(f"Feature names loaded from {feature_names_path}")
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        
        Args:
            input_data (dict or pd.DataFrame): Input data
            
        Returns:
            np.array: Preprocessed data ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select and order features
            input_df = input_df[self.feature_names]
        
        # Scale features if scaler is available
        if self.scaler:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        return input_scaled
    
    def predict(self, input_data):
        """
        Make prediction on input data
        
        Args:
            input_data (dict or pd.DataFrame): Input data
            
        Returns:
            int: Predicted class (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess input
        input_processed = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_processed)
        
        return int(prediction[0])
    
    def predict_proba(self, input_data):
        """
        Get prediction probabilities
        
        Args:
            input_data (dict or pd.DataFrame): Input data
            
        Returns:
            dict: Probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Preprocess input
        input_processed = self.preprocess_input(input_data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(input_processed)[0]
        
        return {
            'No Alzheimer': float(probabilities[0]),
            'Alzheimer': float(probabilities[1])
        }
    
    def predict_with_details(self, input_data):
        """
        Make prediction with detailed information
        
        Args:
            input_data (dict or pd.DataFrame): Input data
            
        Returns:
            dict: Prediction results with details
        """
        # Get prediction
        prediction = self.predict(input_data)
        
        # Get probabilities if available
        try:
            proba_dict = self.predict_proba(input_data)
            probabilities = [proba_dict['No Alzheimer'], proba_dict['Alzheimer']]
        except:
            probabilities = [0.5, 0.5]
        
        # Prepare result
        result = {
            'prediction': prediction,
            'diagnosis': prediction,  # 0 or 1
            'probabilities': probabilities,  # [prob_no_alzheimer, prob_alzheimer]
            'confidence': max(probabilities)
        }
        
        return result
    
    def batch_predict(self, input_data_list):
        """
        Make predictions on multiple inputs
        
        Args:
            input_data_list (list): List of input data dictionaries
            
        Returns:
            list: List of predictions
        """
        predictions = []
        
        for input_data in input_data_list:
            pred = self.predict_with_details(input_data)
            predictions.append(pred)
        
        return predictions


def load_predictor(models_dir='../models/saved_models', 
                   processed_dir='../data/processed',
                   model_name='random_forest_model.pkl'):
    """
    Convenience function to load a complete predictor
    
    Args:
        models_dir (str): Directory containing saved models
        processed_dir (str): Directory containing preprocessing artifacts
        model_name (str): Name of the model file
        
    Returns:
        AlzheimerPredictor: Loaded predictor
    """
    model_path = os.path.join(models_dir, model_name)
    scaler_path = os.path.join(processed_dir, 'scaler.pkl')
    feature_names_path = os.path.join(processed_dir, 'feature_names.pkl')
    
    predictor = AlzheimerPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        feature_names_path=feature_names_path
    )
    
    return predictor


if __name__ == "__main__":
    # Example usage
    print("Prediction Module")
    print("Import this module to use AlzheimerPredictor class")
    
    # Example prediction
    # predictor = load_predictor()
    # sample_data = {
    #     'Age': 75,
    #     'Gender': 1,
    #     'BMI': 25.5,
    #     # ... other features
    # }
    # result = predictor.predict_with_details(sample_data)
    # print(result)
