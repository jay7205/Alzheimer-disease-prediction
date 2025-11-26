"""
Data Preprocessing Module
Handles data loading, cleaning, and preprocessing for Alzheimer's disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Loading data
    - Handling missing values
    - Encoding categorical variables
    - Feature scaling
    - Train-test split
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the DataPreprocessor
        
        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, data_path=None):
        """
        Load data from CSV file
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Data path must be provided")
            
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent')
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        print(f"\nHandling missing values with strategy: {strategy}")
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found {missing_counts.sum()} missing values")
            
            # Separate numeric and categorical columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            
            # Impute numeric columns
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy=strategy)
                self.data[numeric_cols] = numeric_imputer.fit_transform(self.data[numeric_cols])
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                self.data[categorical_cols] = cat_imputer.fit_transform(self.data[categorical_cols])
                
            print("Missing values handled successfully!")
        else:
            print("No missing values found!")
            
        return self.data
    
    def encode_categorical_variables(self, columns=None):
        """
        Encode categorical variables using Label Encoding
        
        Args:
            columns (list): List of columns to encode. If None, auto-detect
            
        Returns:
            pd.DataFrame: DataFrame with encoded variables
        """
        print("\nEncoding categorical variables...")
        
        if columns is None:
            # Auto-detect categorical columns
            columns = self.data.select_dtypes(include=['object']).columns.tolist()
            
        for col in columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded: {col}")
                
        print(f"Encoded {len(columns)} categorical variables")
        return self.data
    
    def prepare_features_target(self, target_column='Diagnosis', drop_columns=None):
        """
        Separate features and target variable
        
        Args:
            target_column (str): Name of the target column
            drop_columns (list): Additional columns to drop
            
        Returns:
            tuple: (X, y) features and target
        """
        print(f"\nPreparing features and target...")
        print(f"Target column: {target_column}")
        
        # Columns to drop
        cols_to_drop = [target_column]
        if drop_columns:
            cols_to_drop.extend(drop_columns)
            
        # Remove columns that don't exist
        cols_to_drop = [col for col in cols_to_drop if col in self.data.columns]
        
        # Separate features and target
        X = self.data.drop(columns=cols_to_drop)
        y = self.data[target_column]
        
        self.feature_names = X.columns.tolist()
        print(f"Features: {X.shape[1]} columns")
        print(f"Target: {y.shape[0]} samples")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self, X_train=None, X_test=None):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        print("\nScaling features...")
        
        if X_train is None:
            X_train = self.X_train
        if X_test is None:
            X_test = self.X_test
            
        # Fit scaler on training data and transform both sets
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("Features scaled successfully!")
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, target_column='Diagnosis', drop_columns=None, 
                           test_size=0.2, scale=True):
        """
        Complete preprocessing pipeline
        
        Args:
            target_column (str): Name of target column
            drop_columns (list): Columns to drop
            test_size (float): Test set proportion
            scale (bool): Whether to scale features
            
        Returns:
            dict: Dictionary containing all processed data
        """
        print("="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode categorical variables
        self.encode_categorical_variables()
        
        # Prepare features and target
        X, y = self.prepare_features_target(target_column, drop_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # Scale features if requested
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test)
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names
        }
    
    def save_processed_data(self, output_dir='../data/processed'):
        """
        Save processed data to files
        
        Args:
            output_dir (str): Directory to save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save datasets
        self.X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        self.X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        self.y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        self.y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # Save scaler and encoders
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{output_dir}/label_encoders.pkl")
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")
        
        print("Processed data saved successfully!")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    preprocessor.load_data("../data/raw/alzheimers_disease_data.csv")
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(
        target_column='Diagnosis',
        drop_columns=['PatientID', 'DoctorInCharge'],
        test_size=0.2,
        scale=True
    )
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\nPreprocessing complete!")
