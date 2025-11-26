"""
Model Training Module
Handles training, evaluation, and comparison of multiple ML models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Handles training and evaluation of multiple ML models
    """
    
    def __init__(self):
        """Initialize the ModelTrainer"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Initialize all models with default parameters
        
        Returns:
            dict: Dictionary of initialized models
        """
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a single model
        
        Args:
            model_name (str): Name of the model
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        print(f"{model_name} training completed!")
        return model
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['ROC-AUC'] = None
        
        # Store confusion matrix
        metrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            pd.DataFrame: Results for all models
        """
        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        results_list = []
        
        for model_name in self.models.keys():
            # Train model
            model = self.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(model, model_name, X_test, y_test)
            results_list.append(metrics)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics
            }
        
        # Create results dataframe
        results_df = pd.DataFrame(results_list)
        results_df = results_df.drop('Confusion Matrix', axis=1)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Identify best model
        best_idx = results_df['Accuracy'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
        
        return results_df
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """
        Perform cross-validation on a model
        
        Args:
            model_name (str): Name of the model
            X: Features
            y: Target
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation on {model_name}...")
        
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        results = {
            'Model': model_name,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'CV Scores': cv_scores
        }
        
        print(f"CV Accuracy: {results['CV Mean']:.4f} (+/- {results['CV Std']:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv=3):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_name (str): Name of the model
            X_train: Training features
            y_train: Training target
            param_grid (dict): Parameter grid for tuning
            cv (int): Number of folds
            
        Returns:
            Best model and parameters
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        print(f"Parameter grid: {param_grid}")
        
        model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def plot_confusion_matrix(self, model_name, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            model_name (str): Name of the model
            y_test: True labels
            y_pred: Predicted labels
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        return plt
    
    def plot_roc_curve(self, model_name, y_test, y_pred_proba, save_path=None):
        """
        Plot ROC curve
        
        Args:
            model_name (str): Name of the model
            y_test: True labels
            y_pred_proba: Predicted probabilities
            save_path (str): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.tight_layout()
        return plt
    
    def save_model(self, model_name=None, save_path='../models/saved_models'):
        """
        Save trained model
        
        Args:
            model_name (str): Name of model to save (default: best model)
            save_path (str): Directory to save the model
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.results[model_name]['model']
        
        # Save model
        model_filename = f"{save_path}/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_filename)
        
        print(f"\nModel saved: {model_filename}")
        
        return model_filename
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def get_classification_report(self, model_name, y_test, y_pred):
        """
        Get detailed classification report
        
        Args:
            model_name (str): Name of the model
            y_test: True labels
            y_pred: Predicted labels
            
        Returns:
            str: Classification report
        """
        print(f"\nClassification Report - {model_name}")
        print("="*60)
        report = classification_report(y_test, y_pred)
        print(report)
        return report


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("Import this module to use ModelTrainer class")
