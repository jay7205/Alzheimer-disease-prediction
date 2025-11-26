/**
 * Alzheimer's Disease Prediction - Frontend JavaScript
 * Handles form submission, API calls, and result visualization
 */

// DOM Elements
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('resultsSection');
const loadingState = document.getElementById('loadingState');
const resultsContent = document.getElementById('resultsContent');
const submitBtn = document.getElementById('submitBtn');

// Form submission handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Show results section with loading state
    resultsSection.style.display = 'block';
    loadingState.style.display = 'block';
    resultsContent.style.display = 'none';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.style.opacity = '0.6';
    submitBtn.querySelector('.btn-text').textContent = 'Analyzing...';

    // Collect form data
    const formData = new FormData(form);
    const patientData = {};

    for (let [key, value] of formData.entries()) {
        // Convert to appropriate type
        patientData[key] = isNaN(value) ? value : parseFloat(value);
    }

    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });

        const result = await response.json();

        if (result.success) {
            // Display results after a short delay for better UX
            setTimeout(() => {
                displayResults(result);
            }, 1000);
        } else {
            showError(result.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the server. Please try again.');
    } finally {
        // Re-enable submit button
        submitBtn.disabled = false;
        submitBtn.style.opacity = '1';
        submitBtn.querySelector('.btn-text').textContent = 'Analyze Patient Data';
    }
});

/**
 * Display prediction results
 */
function displayResults(result) {
    const prediction = result.prediction;

    // Hide loading, show results
    loadingState.style.display = 'none';
    resultsContent.style.display = 'block';

    // Update diagnosis card
    const diagnosisCard = document.getElementById('diagnosisCard');
    const diagnosisIcon = document.getElementById('diagnosisIcon');
    const diagnosisText = document.getElementById('diagnosisText');
    const riskLevel = document.getElementById('riskLevel');

    if (prediction.diagnosis === 1) {
        diagnosisCard.classList.add('positive');
        diagnosisCard.classList.remove('negative');
        diagnosisIcon.textContent = '⚠️';
        diagnosisText.textContent = prediction.diagnosis_label;
        diagnosisText.classList.add('positive');
        diagnosisText.classList.remove('negative');
    } else {
        diagnosisCard.classList.add('negative');
        diagnosisCard.classList.remove('positive');
        diagnosisIcon.textContent = '✅';
        diagnosisText.textContent = prediction.diagnosis_label;
        diagnosisText.classList.add('negative');
        diagnosisText.classList.remove('positive');
    }

    riskLevel.textContent = result.risk_level;

    // Update confidence gauge
    updateGauge(prediction.confidence);

    // Update probability bars
    updateProbabilities(
        prediction.probability_no_alzheimers,
        prediction.probability_alzheimers
    );

    // Update recommendations
    updateRecommendations(result.recommendations);

    // Animate entrance
    resultsContent.style.animation = 'fadeIn 0.5s ease';
}

/**
 * Update confidence gauge
 */
function updateGauge(confidence) {
    const gaugeValue = document.getElementById('gaugeValue');
    const gaugeFill = document.getElementById('gaugeFill');

    const percentage = (confidence * 100).toFixed(1);
    gaugeValue.textContent = `${percentage}%`;

    // Calculate stroke-dashoffset (251.2 is the total path length)
    const totalLength = 251.2;
    const offset = totalLength - (totalLength * confidence);

    // Animate the gauge
    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = offset;
    }, 100);

    // Change color based on confidence
    if (confidence >= 0.9) {
        gaugeFill.style.stroke = '#ff6b6b'; // Red for high confidence
    } else if (confidence >= 0.7) {
        gaugeFill.style.stroke = '#ffa500'; // Orange for moderate
    } else {
        gaugeFill.style.stroke = '#00ff88'; // Green for low
    }
}

/**
 * Update probability bars
 */
function updateProbabilities(probNegative, probPositive) {
    const probNegativeBar = document.getElementById('probNegative');
    const probPositiveBar = document.getElementById('probPositive');
    const probNegativeValue = document.getElementById('probNegativeValue');
    const probPositiveValue = document.getElementById('probPositiveValue');

    const negPercentage = (probNegative * 100).toFixed(1);
    const posPercentage = (probPositive * 100).toFixed(1);

    // Animate bars
    setTimeout(() => {
        probNegativeBar.style.width = `${negPercentage}%`;
        probPositiveBar.style.width = `${posPercentage}%`;
    }, 200);

    probNegativeValue.textContent = `${negPercentage}%`;
    probPositiveValue.textContent = `${posPercentage}%`;
}

/**
 * Update recommendations list
 */
function updateRecommendations(recommendations) {
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';

    recommendations.forEach((rec, index) => {
        const li = document.createElement('li');
        li.textContent = rec;
        li.style.animation = `fadeIn 0.5s ease ${index * 0.1}s both`;
        recommendationsList.appendChild(li);
    });
}

/**
 * Show error message
 */
function showError(message) {
    loadingState.style.display = 'none';
    resultsContent.style.display = 'block';
    resultsContent.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
            <h4 style="color: #ff6b6b; margin-bottom: 1rem;">Error</h4>
            <p style="color: #b4b4c8;">${message}</p>
        </div>
    `;
}

/**
 * Auto-fill form with sample data (for testing)
 */
function fillSampleData() {
    const sampleData = {
        Age: 75,
        Gender: 1,
        Ethnicity: 0,
        EducationLevel: 2,
        BMI: 25.5,
        Smoking: 0,
        AlcoholConsumption: 2,
        PhysicalActivity: 5.5,
        DietQuality: 7.2,
        SleepQuality: 6.8,
        FamilyHistoryAlzheimers: 1,
        CardiovascularDisease: 0,
        Diabetes: 0,
        Depression: 0,
        HeadInjury: 0,
        Hypertension: 1,
        SystolicBP: 135,
        DiastolicBP: 85,
        CholesterolTotal: 210,
        CholesterolLDL: 130,
        CholesterolHDL: 55,
        CholesterolTriglycerides: 150,
        MMSE: 22.5,
        FunctionalAssessment: 6.5,
        MemoryComplaints: 1,
        BehavioralProblems: 0,
        ADL: 7.2,
        Confusion: 1,
        Disorientation: 0,
        PersonalityChanges: 0,
        DifficultyCompletingTasks: 1,
        Forgetfulness: 1
    };

    for (let [key, value] of Object.entries(sampleData)) {
        const input = document.getElementById(key);
        if (input) {
            input.value = value;
        }
    }
}

// Add keyboard shortcut for sample data (Ctrl+Shift+S)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        fillSampleData();
        console.log('Sample data filled!');
    }
});

// Form validation feedback
const inputs = form.querySelectorAll('input, select');
inputs.forEach(input => {
    input.addEventListener('invalid', (e) => {
        e.preventDefault();
        input.style.borderColor = '#ff6b6b';
        setTimeout(() => {
            input.style.borderColor = '';
        }, 2000);
    });

    input.addEventListener('input', () => {
        if (input.checkValidity()) {
            input.style.borderColor = '#00ff88';
            setTimeout(() => {
                input.style.borderColor = '';
            }, 1000);
        }
    });
});

console.log('🧠 Alzheimer\'s AI Prediction System loaded');
console.log('💡 Tip: Press Ctrl+Shift+S to fill sample data for testing');
