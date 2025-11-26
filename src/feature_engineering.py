"""
Feature Engineering Module
Handles feature creation, selection, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib


class FeatureEngineer:
    """
    Handles feature engineering tasks including:
    - Feature creation
    - Feature selection
    - Dimensionality reduction
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer"""
        self.selected_features = None
        self.feature_importance = None
        self.pca = None
        
    def create_interaction_features(self, df, feature_pairs):
        """
        Create interaction features by multiplying feature pairs
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_pairs (list): List of tuples containing feature pairs
            
        Returns:
            pd.DataFrame: DataFrame with new interaction features
        """
        print("\nCreating interaction features...")
        df_new = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                new_feature_name = f"{feat1}_x_{feat2}"
                df_new[new_feature_name] = df[feat1] * df[feat2]
                print(f"  Created: {new_feature_name}")
                
        print(f"Created {len(feature_pairs)} interaction features")
        return df_new
    
    def create_age_groups(self, df, age_column='Age'):
        """
        Create age group categories
        
        Args:
            df (pd.DataFrame): Input dataframe
            age_column (str): Name of age column
            
        Returns:
            pd.DataFrame: DataFrame with age group feature
        """
        print("\nCreating age groups...")
        df_new = df.copy()
        
        if age_column in df.columns:
            # Fill any NaN values first
            age_values = df[age_column].fillna(df[age_column].median())
            df_new['AgeGroup'] = pd.cut(
                age_values,
                bins=[0, 65, 75, 85, 100],
                labels=[0, 1, 2, 3]  # Young-old, Middle-old, Old-old, Very-old
            )
            # Convert to int, handling any remaining NaN
            df_new['AgeGroup'] = df_new['AgeGroup'].fillna(1).astype(int)
            print("  Created: AgeGroup (0: <65, 1: 65-75, 2: 75-85, 3: 85+)")
            
        return df_new
    
    def create_health_risk_score(self, df):
        """
        Create a composite health risk score
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with health risk score
        """
        print("\nCreating health risk score...")
        df_new = df.copy()
        
        risk_factors = []
        
        # Add risk factors if columns exist
        if 'Smoking' in df.columns:
            risk_factors.append(df['Smoking'])
        if 'Diabetes' in df.columns:
            risk_factors.append(df['Diabetes'])
        if 'Hypertension' in df.columns:
            risk_factors.append(df['Hypertension'])
        if 'CardiovascularDisease' in df.columns:
            risk_factors.append(df['CardiovascularDisease'])
        if 'Depression' in df.columns:
            risk_factors.append(df['Depression'])
            
        if risk_factors:
            df_new['HealthRiskScore'] = sum(risk_factors)
            print(f"  Created: HealthRiskScore (sum of {len(risk_factors)} risk factors)")
            
        return df_new
    
    def create_cognitive_score(self, df):
        """
        Create a composite cognitive impairment score
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with cognitive score
        """
        print("\nCreating cognitive impairment score...")
        df_new = df.copy()
        
        cognitive_factors = []
        
        # Add cognitive factors if columns exist
        if 'MemoryComplaints' in df.columns:
            cognitive_factors.append(df['MemoryComplaints'])
        if 'Confusion' in df.columns:
            cognitive_factors.append(df['Confusion'])
        if 'Disorientation' in df.columns:
            cognitive_factors.append(df['Disorientation'])
        if 'Forgetfulness' in df.columns:
            cognitive_factors.append(df['Forgetfulness'])
        if 'DifficultyCompletingTasks' in df.columns:
            cognitive_factors.append(df['DifficultyCompletingTasks'])
            
        if cognitive_factors:
            df_new['CognitiveImpairmentScore'] = sum(cognitive_factors)
            print(f"  Created: CognitiveImpairmentScore (sum of {len(cognitive_factors)} factors)")
            
        return df_new
    
    def select_features_univariate(self, X, y, k=20):
        """
        Select top k features using univariate statistical tests
        
        Args:
            X: Features
            y: Target
            k (int): Number of features to select
            
        Returns:
            list: Selected feature names
        """
        print(f"\nSelecting top {k} features using univariate tests...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get feature scores
        scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        print(f"Selected features: {selected_features[:5]}... (showing top 5)")
        
        self.selected_features = selected_features
        return selected_features, scores
    
    def select_features_rfe(self, X, y, n_features=20):
        """
        Select features using Recursive Feature Elimination
        
        Args:
            X: Features
            y: Target
            n_features (int): Number of features to select
            
        Returns:
            list: Selected feature names
        """
        print(f"\nSelecting {n_features} features using RFE...")
        
        # Use Random Forest as the estimator
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        print(f"Selected features: {selected_features[:5]}... (showing top 5)")
        
        self.selected_features = selected_features
        return selected_features
    
    def get_feature_importance(self, X, y, top_n=20):
        """
        Get feature importance using Random Forest
        
        Args:
            X: Features
            y: Target
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        print(f"\nCalculating feature importance...")
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"Top {min(top_n, len(importance_df))} important features:")
        print(importance_df.head(top_n))
        
        self.feature_importance = importance_df
        return importance_df
    
    def apply_pca(self, X_train, X_test, n_components=0.95):
        """
        Apply PCA for dimensionality reduction
        
        Args:
            X_train: Training features
            X_test: Test features
            n_components: Number of components or variance to retain
            
        Returns:
            tuple: (X_train_pca, X_test_pca)
        """
        print(f"\nApplying PCA (n_components={n_components})...")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        print(f"Original features: {X_train.shape[1]}")
        print(f"PCA components: {X_train_pca.shape[1]}")
        print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return X_train_pca, X_test_pca
    
    def engineer_features_pipeline(self, X_train, X_test, y_train):
        """
        Complete feature engineering pipeline
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            
        Returns:
            dict: Dictionary with engineered features
        """
        print("="*60)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Create new features
        X_train_new = self.create_age_groups(X_train)
        X_test_new = self.create_age_groups(X_test)
        
        X_train_new = self.create_health_risk_score(X_train_new)
        X_test_new = self.create_health_risk_score(X_test_new)
        
        X_train_new = self.create_cognitive_score(X_train_new)
        X_test_new = self.create_cognitive_score(X_test_new)
        
        # Get feature importance
        importance_df = self.get_feature_importance(X_train_new, y_train)
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETED!")
        print("="*60)
        
        return {
            'X_train': X_train_new,
            'X_test': X_test_new,
            'feature_importance': importance_df
        }
    
    def save_feature_engineer(self, output_path='../models/feature_engineer.pkl'):
        """
        Save feature engineer object
        
        Args:
            output_path (str): Path to save the object
        """
        joblib.dump(self, output_path)
        print(f"\nFeature engineer saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Import this module to use FeatureEngineer class")
