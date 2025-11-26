# Quick Start Guide - Alzheimer's Disease Prediction

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
cd alzheimers_prediction
pip install -r requirements.txt
```

### Step 2: Run the Notebooks in Order

#### Option A: Using Jupyter Notebook
```bash
jupyter notebook
```
Then open notebooks in this order:
1. `notebooks/01_eda.ipynb` - Explore the data
2. `notebooks/02_preprocessing.ipynb` - Preprocess the data
3. `notebooks/03_modeling.ipynb` - Train models

#### Option B: Using Python Scripts

**Preprocessing:**
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
preprocessor.load_data('data/raw/alzheimers_disease_data.csv')
data = preprocessor.preprocess_pipeline(
    target_column='Diagnosis',
    drop_columns=['PatientID', 'DoctorInCharge']
)
preprocessor.save_processed_data('data/processed')
```

**Model Training:**
```python
from src.model_training import ModelTrainer
import pandas as pd

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Train models
trainer = ModelTrainer()
trainer.initialize_models()
results = trainer.train_all_models(X_train, y_train, X_test, y_test)
trainer.save_model()
```

**Making Predictions:**
```python
from src.prediction import load_predictor

predictor = load_predictor()
result = predictor.predict_with_details(patient_data)
print(result)
```

## 📊 What Each Notebook Does

### 01_eda.ipynb
- Loads and explores the dataset
- Analyzes distributions and correlations
- Identifies missing values and outliers
- Creates visualizations

### 02_preprocessing.ipynb
- Handles missing values
- Encodes categorical variables
- Scales features
- Splits data into train/test sets
- Saves processed data

### 03_modeling.ipynb
- Applies feature engineering
- Trains 6 ML models
- Compares model performance
- Evaluates best model
- Saves trained model

## 🎯 Expected Workflow

```
1. EDA (01_eda.ipynb)
   ↓
2. Preprocessing (02_preprocessing.ipynb)
   ↓
3. Modeling (03_modeling.ipynb)
   ↓
4. Deployment (Flask app)
```

## 📁 File Locations

- **Raw Data**: `data/raw/alzheimers_disease_data.csv`
- **Processed Data**: `data/processed/` (created after preprocessing)
- **Saved Models**: `models/saved_models/` (created after training)
- **Python Modules**: `src/`
- **Notebooks**: `notebooks/`

## 🔧 Troubleshooting

### Import Errors
If you get import errors in notebooks, add this at the top:
```python
import sys
sys.path.append('../src')
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Data Not Found
Make sure you're running from the `alzheimers_prediction` directory.

## 📈 Next Steps After Training

1. Review model performance in `03_modeling.ipynb`
2. Use the best model for predictions
3. Deploy as a web application (coming soon)

---

**Ready to start? Open `notebooks/01_eda.ipynb` and begin exploring!**
