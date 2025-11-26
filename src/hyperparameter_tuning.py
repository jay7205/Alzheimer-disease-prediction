"""
Hyperparameter Tuning Module
Optimizes model performance using GridSearchCV and RandomizedSearchCV
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from datetime import datetime
import json


class HyperparameterTuner:
    """
    Hyperparameter tuning for machine learning models
    """
    
    def __init__(self):
        self.param_grids = {}
        self.tuned_models = {}
        self.tuning_results = {}
        self.best_params = {}
        
    def define_parameter_grids(self):
        """
        Define parameter grids for each model
        """
        # Gradient Boosting parameter grid
        self.param_grids['gradient_boosting'] = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Random Forest parameter grid
        self.param_grids['random_forest'] = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        # XGBoost parameter grid
        self.param_grids['xgboost'] = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        print("[INFO] Parameter grids defined for 3 models")
        return self.param_grids
    
    def define_randomized_search_params(self):
        """
        Define parameter distributions for RandomizedSearchCV (faster)
        """
        # Gradient Boosting
        self.param_grids['gradient_boosting_random'] = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Random Forest
        self.param_grids['random_forest_random'] = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        # XGBoost
        self.param_grids['xgboost_random'] = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.3, 0.5],
            'reg_lambda': [1, 1.2, 1.5, 2]
        }
        
        print("[INFO] Randomized search parameters defined")
        return self.param_grids
    
    def grid_search_tuning(self, model_name, X_train, y_train, cv=5, n_jobs=-1):
        """
        Perform Grid Search hyperparameter tuning
        
        Args:
            model_name: Name of the model ('gradient_boosting', 'random_forest', 'xgboost')
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        print(f"\n{'='*80}")
        print(f"GRID SEARCH TUNING: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Initialize base model
        if model_name == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
        elif model_name == 'xgboost':
            base_model = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get parameter grid
        param_grid = self.param_grids.get(model_name, {})
        
        print(f"[INFO] Parameter grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations")
        print(f"[INFO] Cross-validation folds: {cv}")
        print(f"[INFO] This may take several minutes...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.tuned_models[model_name] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_
        self.tuning_results[model_name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"\n[SUCCESS] Grid search completed!")
        print(f"   Best CV Score: {grid_search.best_score_:.4f}")
        print(f"\n   Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def randomized_search_tuning(self, model_name, X_train, y_train, n_iter=100, cv=5, n_jobs=-1):
        """
        Perform Randomized Search hyperparameter tuning (faster than grid search)
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        print(f"\n{'='*80}")
        print(f"RANDOMIZED SEARCH TUNING: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Initialize base model
        if model_name == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            param_dist = self.param_grids.get('gradient_boosting_random', {})
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
            param_dist = self.param_grids.get('random_forest_random', {})
        elif model_name == 'xgboost':
            base_model = XGBClassifier(random_state=42, eval_metric='logloss')
            param_dist = self.param_grids.get('xgboost_random', {})
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"[INFO] Sampling {n_iter} parameter combinations")
        print(f"[INFO] Cross-validation folds: {cv}")
        print(f"[INFO] This may take several minutes...")
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        # Store results
        self.tuned_models[f"{model_name}_random"] = random_search.best_estimator_
        self.best_params[f"{model_name}_random"] = random_search.best_params_
        self.tuning_results[f"{model_name}_random"] = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }
        
        print(f"\n[SUCCESS] Randomized search completed!")
        print(f"   Best CV Score: {random_search.best_score_:.4f}")
        print(f"\n   Best Parameters:")
        for param, value in random_search.best_params_.items():
            print(f"      {param}: {value}")
        
        return random_search.best_estimator_, random_search.best_params_
    
    def evaluate_tuned_model(self, model_name, model, X_test, y_test):
        """
        Evaluate tuned model on test set
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n[EVALUATION] {model_name}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_baseline_vs_tuned(self, baseline_results, tuned_results):
        """
        Compare baseline and tuned model performance
        """
        print(f"\n{'='*80}")
        print("BASELINE vs TUNED MODEL COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame({
            'Model': [],
            'Type': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'ROC-AUC': [],
            'Improvement': []
        })
        
        for model_name in tuned_results.keys():
            base_name = model_name.replace('_random', '')
            if base_name in baseline_results:
                baseline = baseline_results[base_name]
                tuned = tuned_results[model_name]
                
                improvement = (tuned['accuracy'] - baseline['accuracy']) * 100
                
                # Add baseline row
                comparison_df = pd.concat([comparison_df, pd.DataFrame({
                    'Model': [base_name],
                    'Type': ['Baseline'],
                    'Accuracy': [f"{baseline['accuracy']:.4f}"],
                    'Precision': [f"{baseline['precision']:.4f}"],
                    'Recall': [f"{baseline['recall']:.4f}"],
                    'F1-Score': [f"{baseline['f1_score']:.4f}"],
                    'ROC-AUC': [f"{baseline['roc_auc']:.4f}"],
                    'Improvement': ['-']
                })], ignore_index=True)
                
                # Add tuned row
                comparison_df = pd.concat([comparison_df, pd.DataFrame({
                    'Model': [base_name],
                    'Type': ['Tuned'],
                    'Accuracy': [f"{tuned['accuracy']:.4f}"],
                    'Precision': [f"{tuned['precision']:.4f}"],
                    'Recall': [f"{tuned['recall']:.4f}"],
                    'F1-Score': [f"{tuned['f1_score']:.4f}"],
                    'ROC-AUC': [f"{tuned['roc_auc']:.4f}"],
                    'Improvement': [f"+{improvement:.2f}%"]
                })], ignore_index=True)
        
        print(comparison_df.to_string(index=False))
        return comparison_df
    
    def save_tuned_model(self, model_name, model, save_dir='models/tuned_models'):
        """
        Save tuned model and parameters
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f'{model_name}_tuned.pkl')
        joblib.dump(model, model_path)
        
        # Save parameters
        params_path = os.path.join(save_dir, f'{model_name}_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.best_params.get(model_name, {}), f, indent=4)
        
        print(f"\n[SAVED] Tuned model: {model_path}")
        print(f"[SAVED] Parameters: {params_path}")
        
        return model_path
    
    def save_tuning_results(self, save_dir='models/tuned_models'):
        """
        Save all tuning results to file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        results_path = os.path.join(save_dir, 'tuning_results.json')
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, results in self.tuning_results.items():
            json_results[model_name] = {
                'best_score': float(results['best_score']),
                'best_params': results['best_params'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"\n[SAVED] Tuning results: {results_path}")
        return results_path
