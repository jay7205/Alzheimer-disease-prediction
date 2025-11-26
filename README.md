## 📋 Project Overview

A complete end-to-end machine learning system for predicting Alzheimer's disease using patient health data. Features hyperparameter-tuned models achieving **95.12% accuracy**, a modern web application with premium UI, and comprehensive data analysis pipelines.

### ✨ Key Features

- 🎯 **95.12% Prediction Accuracy** with tuned Random Forest model
- 🌐 **Modern Web Application** with Flask backend and glassmorphism UI
- 📊 **Comprehensive ML Pipeline** from EDA to deployment
- 🔬 **Hyperparameter Optimization** using RandomizedSearchCV
- 📈 **Interactive Visualizations** and real-time predictions
- 🚀 **Production Ready** with deployment guides

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest (Tuned)** 🏆 | **95.12%** | **95.11%** | **95.12%** | **95.10%** | **93.91%** |
| Gradient Boosting (Tuned) | 94.88% | 94.87% | 94.88% | 94.88% | 94.77% |
| XGBoost (Tuned) | 94.42% | 94.40% | 94.42% | 94.40% | 94.16% |

*Improvement of +0.93% over baseline through hyperparameter tuning*

## 📊 Dataset

- **Source**: Alzheimer's Disease Patient Data
- **Samples**: 2,149 patients
- **Features**: 35 health indicators
  - Demographics (Age, Gender, Ethnicity, Education)
  - Health Metrics (BMI, Blood Pressure, Cholesterol)
  - Lifestyle Factors (Smoking, Alcohol, Physical Activity)
  - Medical History (Diabetes, Cardiovascular Disease, Depression)
  - Cognitive Assessments (MMSE, Functional Assessment, ADL)
  - Symptoms (Memory Complaints, Confusion, Disorientation)

## 🗂️ Project Structure

```
alzheimers_prediction/
├── app/                          # Web Application
│   ├── app.py                    # Flask backend
│   ├── templates/                # HTML templates
│   │   └── index.html           # Main web interface
│   └── static/                   # CSS, JavaScript
│       ├── style.css            # Premium glassmorphism styling
│       └── script.js            # Frontend logic
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Processed data & artifacts
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data Preprocessing
│   ├── 03_modeling.ipynb         # Model Training
│   └── 04_hyperparameter_tuning.ipynb  # Hyperparameter Optimization
├── src/                          # Source Code Modules
│   ├── data_preprocessing.py     # Data preprocessing
│   ├── feature_engineering.py    # Feature engineering
│   ├── model_training.py         # Model training
│   ├── hyperparameter_tuning.py  # Hyperparameter tuning
│   └── prediction.py             # Prediction module
├── models/
│   ├── saved_models/             # Baseline trained models
│   └── tuned_models/             # Hyperparameter-tuned models
├── tests/                        # Unit tests
├── DEPLOYMENT.md                 # Deployment guide
├── RESULTS.md                    # Detailed results
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd alzheimers_prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Web Application

```bash
cd app
python app.py
```

Access the application at: **http://localhost:5000**

### Quick Test

1. Open the web interface
2. Press `Ctrl+Shift+S` to auto-fill sample data
3. Click "Analyze Patient Data"
4. View prediction results with confidence gauge

## 📚 Usage

### 1. Complete ML Pipeline

Run the entire pipeline from preprocessing to model training:

```bash
python run_pipeline.py
```

### 2. Hyperparameter Tuning

Open the Jupyter notebook for interactive tuning:

```bash
jupyter notebook notebooks/04_hyperparameter_tuning.ipynb
```

Or run the Python script:

```bash
python run_hyperparameter_tuning.py
```

### 3. Making Predictions (Python)

```python
from src.prediction import AlzheimerPredictor

# Load model
predictor = AlzheimerPredictor()
predictor.load_model('models/tuned_models/random_forest_tuned.pkl')
predictor.load_scaler('data/processed/scaler.pkl')
predictor.load_feature_names('data/processed/feature_names.pkl')

# Make prediction
patient_data = {
    'Age': 75,
    'Gender': 1,
    'BMI': 25.5,
    'MMSE': 22.5,
    # ... other 31 features
}

result = predictor.predict_with_details(patient_data)
print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 4. API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_patient.json
```

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling (StandardScaler)
- Train-test split (80/20)

### 2. Feature Engineering
- Age grouping
- Health risk score calculation
- Cognitive impairment score
- Feature importance analysis

### 3. Model Training
Six algorithms compared:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- XGBoost

### 4. Hyperparameter Tuning
- RandomizedSearchCV with 50 iterations
- 5-fold cross-validation
- Optimized top 3 models
- Best model: Random Forest (95.12% accuracy)

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Cross-validation
- Classification reports

## 🌐 Web Application

### Features

- **Modern UI**: Glassmorphism design with dark theme
- **Interactive Form**: 35 patient health indicators
- **Real-time Predictions**: Instant AI-powered diagnosis
- **Confidence Gauge**: Animated SVG visualization
- **Probability Bars**: Visual representation of prediction probabilities
- **Personalized Recommendations**: Health advice based on diagnosis
- **Responsive Design**: Works on desktop, tablet, and mobile

### Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **ML Model**: Scikit-learn Random Forest (tuned)
- **Visualization**: SVG-based gauges and charts

## 📈 Key Insights

### Most Important Features

1. **MMSE** (Mini-Mental State Examination)
2. **Functional Assessment**
3. **Memory Complaints**
4. **ADL** (Activities of Daily Living)
5. **Age**
6. **Cognitive Impairment Score** (engineered feature)
7. **Confusion**
8. **Forgetfulness**
9. **BMI**
10. **Difficulty Completing Tasks**

### Model Improvements

- Hyperparameter tuning improved accuracy by **+0.93%**
- Random Forest outperformed Gradient Boosting after optimization
- Ensemble methods significantly better than linear models
- Cross-validation shows consistent performance (low variance)

## 🛠️ Technologies Used

- **Python 3.10** - Core programming language
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization
- **Flask** - Web framework
- **Jupyter** - Interactive notebooks
- **HTML/CSS/JavaScript** - Web interface

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for complete list:

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
flask==2.3.3
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
```

## 🚀 Deployment

### Local Development

```bash
cd app
python app.py
```

### Production Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed instructions on:
- Heroku deployment
- AWS EC2 setup
- Azure App Service
- Google Cloud Run
- Production configurations
- Security best practices

### Using Gunicorn (Production)

```bash
pip install gunicorn
cd app
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Results

Detailed results available in [`RESULTS.md`](RESULTS.md):
- Complete performance metrics
- Cross-validation results
- Feature importance analysis
- Model comparison charts
- Confusion matrices

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

Test web application:

```bash
cd app
python app.py
# Open http://localhost:5000 in browser
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**JAY**

## 🙏 Acknowledgments

- Alzheimer's Disease Patient Dataset
- Scikit-learn and XGBoost documentation
- Flask framework
- Open source ML community

## 📞 Contact

For questions or feedback, please open an issue in the repository.

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---
