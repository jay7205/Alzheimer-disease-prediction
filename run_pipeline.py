"""
Complete ML Pipeline Runner
Executes the entire machine learning pipeline from preprocessing to model training
"""

import sys
import os
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("ALZHEIMER'S DISEASE PREDICTION - COMPLETE ML PIPELINE")
    print("="*80)
    
    # ============================================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor()
    preprocessor.load_data('data/raw/alzheimers_disease_data.csv')
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(
        target_column='Diagnosis',
        drop_columns=['PatientID', 'DoctorInCharge'],
        test_size=0.2,
        scale=True
    )
    
    # Save processed data
    preprocessor.save_processed_data('data/processed')
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    print(f"\n[SUCCESS] Preprocessing completed!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # ============================================================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features_pipeline(X_train, X_test, y_train)
    
    X_train_eng = engineered_data['X_train']
    X_test_eng = engineered_data['X_test']
    feature_importance = engineered_data['feature_importance']
    
    print(f"\n[SUCCESS] Feature engineering completed!")
    print(f"   New feature count: {X_train_eng.shape[1]}")
    print(f"\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # ============================================================================
    # STEP 3: MODEL TRAINING
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80)
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Train all models
    results_df = trainer.train_all_models(
        X_train_eng, y_train,
        X_test_eng, y_test
    )
    
    print(f"\n[SUCCESS] Model training completed!")
    
    # ============================================================================
    # STEP 4: DETAILED EVALUATION OF BEST MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 4: BEST MODEL EVALUATION")
    print("="*80)
    
    best_model_name = trainer.best_model_name
    best_model = trainer.best_model
    
    print(f"\n[BEST MODEL] {best_model_name}")
    
    # Make predictions
    y_pred = best_model.predict(X_test_eng)
    
    # Get classification report
    trainer.get_classification_report(best_model_name, y_test, y_pred)
    
    # ============================================================================
    # STEP 5: CROSS-VALIDATION
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 5: CROSS-VALIDATION")
    print("="*80)
    
    import numpy as np
    cv_results = trainer.cross_validate_model(
        best_model_name,
        pd.concat([X_train_eng, X_test_eng]),
        np.concatenate([y_train, y_test]),
        cv=5
    )
    
    # ============================================================================
    # STEP 6: SAVE BEST MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING BEST MODEL")
    print("="*80)
    
    model_path = trainer.save_model()
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\nFINAL RESULTS:")
    print(f"   Dataset: 2,150 patients")
    print(f"   Features: {X_train_eng.shape[1]}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Best Model: {best_model_name}")
    print(f"   Model saved to: {model_path}")
    
    print(f"\nMODEL COMPARISON:")
    print(results_df.to_string(index=False))
    
    print(f"\n[SUCCESS] All data and models saved successfully!")
    print(f"   - Processed data: data/processed/")
    print(f"   - Trained model: {model_path}")
    
    print("\n" + "="*80)
    print("Ready for deployment and predictions!")
    print("="*80)
    
    return results_df, best_model_name, model_path

if __name__ == "__main__":
    results, best_model, model_path = main()
