# Deployment Guide - Alzheimer's Disease Prediction Web Application

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

## 🚀 Local Development Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure the trained model exists in one of these locations:
- `models/tuned_models/gradient_boosting_tuned.pkl` (recommended)
- `models/saved_models/gradient_boosting_model.pkl` (fallback)

### 3. Run the Application

```bash
cd app
python app.py
```

The application will start on `http://localhost:5000`

### 4. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 🧪 Testing the Application

### Quick Test with Sample Data

1. Press `Ctrl+Shift+S` in the browser to auto-fill sample patient data
2. Click "Analyze Patient Data"
3. View the prediction results

### Manual Testing

Fill out the form with patient information:
- **Demographics**: Age, Gender, Ethnicity, Education
- **Lifestyle**: BMI, Smoking, Physical Activity, Diet, Sleep
- **Medical History**: Family history, cardiovascular disease, diabetes, etc.
- **Vital Signs**: Blood pressure, cholesterol levels
- **Cognitive Assessment**: MMSE score, functional assessment, ADL
- **Symptoms**: Confusion, disorientation, memory complaints, etc.

## 🌐 Cloud Deployment Options

### Option 1: Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: cd app && python app.py" > Procfile

# Create runtime.txt
echo "python-3.10.12" > runtime.txt

# Deploy
heroku create alzheimers-prediction-app
git push heroku main
```

### Option 2: AWS EC2

1. Launch an EC2 instance (Ubuntu 20.04 LTS)
2. SSH into the instance
3. Install Python and dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```
4. Run the application:
```bash
cd app
python3 app.py
```
5. Configure security group to allow port 5000

### Option 3: Azure App Service

```bash
# Install Azure CLI
az login
az webapp up --name alzheimers-prediction --runtime PYTHON:3.10
```

### Option 4: Google Cloud Run

```bash
# Create Dockerfile
# Build and deploy
gcloud run deploy alzheimers-prediction --source .
```

## 🔒 Production Considerations

### Security

1. **Disable Debug Mode**
   ```python
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```

2. **Use Environment Variables**
   ```python
   import os
   SECRET_KEY = os.environ.get('SECRET_KEY')
   ```

3. **Add HTTPS**
   - Use a reverse proxy (Nginx)
   - Configure SSL certificates (Let's Encrypt)

### Performance

1. **Use Production WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Enable Caching**
   - Cache model predictions for common inputs
   - Use Redis for session management

3. **Add Rate Limiting**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=get_remote_address)
   ```

### Monitoring

1. **Add Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Health Checks**
   - Use the `/health` endpoint for monitoring
   - Set up uptime monitoring (UptimeRobot, Pingdom)

3. **Error Tracking**
   - Integrate Sentry for error tracking
   - Set up alerts for failures

## 📊 API Documentation

### POST /predict

**Request:**
```json
{
  "Age": 75,
  "Gender": 1,
  "BMI": 25.5,
  "MMSE": 22.5,
  // ... other 31 features
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "diagnosis": 1,
    "diagnosis_label": "Alzheimer's Disease",
    "confidence": 0.92,
    "probability_no_alzheimers": 0.08,
    "probability_alzheimers": 0.92
  },
  "risk_level": "High Risk",
  "recommendations": [
    "Consult with a neurologist for comprehensive evaluation",
    "Consider cognitive assessment and brain imaging"
  ]
}
```

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🐛 Troubleshooting

### Model Not Loading

**Error:** `Model not loaded`

**Solution:**
- Verify model file exists in `models/saved_models/` or `models/tuned_models/`
- Check file permissions
- Ensure all preprocessing artifacts are present

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000  # On Mac/Linux
netstat -ano | findstr :5000  # On Windows

# Kill the process or use a different port
python app.py --port 5001
```

### Missing Dependencies

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

## 📝 Environment Variables

Create a `.env` file:

```env
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
MODEL_PATH=models/tuned_models/gradient_boosting_tuned.pkl
PORT=5000
```

## 🔄 Updating the Model

1. Train a new model using `run_pipeline.py` or `run_hyperparameter_tuning.py`
2. Save the model to `models/saved_models/` or `models/tuned_models/`
3. Restart the Flask application
4. The new model will be loaded automatically

## 📱 Mobile Responsiveness

The web interface is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablets (iPad, Android tablets)
- Mobile phones (iOS, Android)

## 🎨 Customization

### Changing Colors

Edit `app/static/style.css`:
```css
:root {
    --accent-purple: #667eea;  /* Change primary color */
    --accent-green: #00ff88;   /* Change success color */
}
```

### Adding Features

1. Update the form in `app/templates/index.html`
2. Modify the prediction logic in `app/app.py`
3. Update the model training pipeline if needed

## 📞 Support

For issues or questions:
- Check the logs: `tail -f app.log`
- Review the error messages in the browser console
- Ensure all dependencies are installed correctly

---

**Last Updated:** November 25, 2025
**Version:** 1.0.0
