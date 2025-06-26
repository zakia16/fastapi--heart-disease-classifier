# Heart Disease Prediction API

A FastAPI-based machine learning API for predicting heart disease based on patient features. This project uses a Logistic Regression model trained on the UCI Heart Disease dataset with 88.52% accuracy.

## Features

- **Single Prediction**: Predict heart disease for individual patients
- **Batch Prediction**: Predict heart disease for multiple patients at once
- **Model Information**: Get details about the trained model
- **Health Check**: Monitor API health and model status
- **Interactive Documentation**: Auto-generated API documentation with Swagger UI

## Model Performance

The Logistic Regression model achieves:
- **Accuracy**: 88.52%
- **Cross-validation**: 83% average accuracy
- **Best performing model** among tested algorithms (Random Forest, XGBoost, SVM, Decision Trees)

## API Endpoints

### 1. Root Endpoint
- **GET** `/` - API information and available endpoints

### 2. Health Check
- **GET** `/health` - Check API health and model status

### 3. Model Information
- **GET** `/model-info` - Get model details and performance metrics

### 4. Single Prediction
- **POST** `/predict` - Predict heart disease for a single patient

### 5. Batch Prediction
- **POST** `/predict-batch` - Predict heart disease for multiple patients

## Input Features

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| `age` | Patient age in years | 29-77 |
| `sex` | Patient gender | 0 = Female, 1 = Male |
| `cp` | Chest pain type | 0-3 |
| `trestbps` | Resting blood pressure (mm Hg) | 94-200 |
| `chol` | Serum cholesterol (mg/dl) | 126-564 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0 = False, 1 = True |
| `restecg` | Resting electrocardiographic results | 0-2 |
| `thalach` | Maximum heart rate achieved | 71-202 |
| `exang` | Exercise induced angina | 0 = No, 1 = Yes |
| `oldpeak` | ST depression induced by exercise | 0.0-6.2 |
| `slope` | Slope of peak exercise ST segment | 0-2 |
| `ca` | Number of major vessels colored by fluoroscopy | 0-4 |
| `thal` | Thalassemia | 0-3 |

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd heart-disease-prediction-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file is present**
   Make sure `heart_disease_model.pkl` is in the project directory.

## Usage

### Running the API

```bash
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### Example API Calls

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 63,
       "sex": 1,
       "cp": 3,
       "trestbps": 145,
       "chol": 233,
       "fbs": 1,
       "restecg": 0,
       "thalach": 150,
       "exang": 0,
       "oldpeak": 2.3,
       "slope": 0,
       "ca": 0,
       "thal": 1
     }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict-batch" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "age": 63,
         "sex": 1,
         "cp": 3,
         "trestbps": 145,
         "chol": 233,
         "fbs": 1,
         "restecg": 0,
         "thalach": 150,
         "exang": 0,
         "oldpeak": 2.3,
         "slope": 0,
         "ca": 0,
         "thal": 1
       },
       {
         "age": 37,
         "sex": 1,
         "cp": 2,
         "trestbps": 130,
         "chol": 250,
         "fbs": 0,
         "restecg": 1,
         "thalach": 187,
         "exang": 0,
         "oldpeak": 3.5,
         "slope": 0,
         "ca": 0,
         "thal": 2
       }
     ]'
```

### Python Client Example

```python
import requests
import json

# API base URL
base_url = "http://localhost:8000"

# Single prediction
patient_data = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

response = requests.post(f"{base_url}/predict", json=patient_data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Confidence: {result['confidence']}")
```

## Model Training

The model was trained using the following process:

1. **Data Preprocessing**: Feature engineering with one-hot encoding for categorical variables
2. **Feature Scaling**: StandardScaler applied to numerical features
3. **Model Selection**: Logistic Regression performed best among multiple algorithms
4. **Cross-validation**: 5-fold cross-validation for robust evaluation
5. **Model Persistence**: Trained model saved using joblib

## API Response Format

### Single Prediction Response
```json
{
  "prediction": 1,
  "probability": 0.85,
  "confidence": "High"
}
```

### Batch Prediction Response
```json
{
  "predictions": [
    {
      "patient_id": 1,
      "prediction": 1,
      "probability": 0.85,
      "confidence": "High"
    }
  ],
  "total_patients": 1
}
```

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request**: Invalid input data
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Prediction errors

## Development

### Project Structure
```
heart-disease-prediction-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── heart_disease_model.pkl # Trained model file
└── Heart_disease_prediction.ipynb # Original Jupyter notebook
```

### Adding New Features

1. **New Models**: Add new model files and update the model loading logic
2. **Additional Endpoints**: Extend the FastAPI app with new routes
3. **Data Validation**: Enhance Pydantic models for better input validation
4. **Authentication**: Add API key or JWT authentication
5. **Database Integration**: Store predictions and patient data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- FastAPI for the excellent web framework
- Scikit-learn for the machine learning algorithms 