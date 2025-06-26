import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            print("✅ Model info retrieved successfully")
            info = response.json()
            print(f"Model Type: {info['model_type']}")
            print(f"Accuracy: {info['accuracy']:.4f}")
            print(f"Features: {len(info['features'])} features")
        else:
            print(f"❌ Model info failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")

def test_single_prediction():
    """Test the single prediction endpoint"""
    print("\nTesting single prediction...")
    
    # Sample patient data
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
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=patient_data)
        if response.status_code == 200:
            print("✅ Single prediction successful")
            result = response.json()
            print(f"Prediction: {result['prediction']} ({'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'})")
            print(f"Probability: {result['probability']:.3f}")
            print(f"Confidence: {result['confidence']}")
        else:
            print(f"❌ Single prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Single prediction error: {e}")

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    # Sample batch data
    batch_data = [
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
        },
        {
            "age": 41,
            "sex": 0,
            "cp": 1,
            "trestbps": 130,
            "chol": 204,
            "fbs": 0,
            "restecg": 0,
            "thalach": 172,
            "exang": 0,
            "oldpeak": 1.4,
            "slope": 2,
            "ca": 0,
            "thal": 2
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
        if response.status_code == 200:
            print("✅ Batch prediction successful")
            result = response.json()
            print(f"Total patients: {result['total_patients']}")
            for pred in result['predictions']:
                print(f"Patient {pred['patient_id']}: {pred['prediction']} "
                      f"({'Heart Disease' if pred['prediction'] == 1 else 'No Heart Disease'}) "
                      f"- Probability: {pred['probability']:.3f}, Confidence: {pred['confidence']}")
        else:
            print(f"❌ Batch prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    # Wait a moment for the API to be ready
    time.sleep(2)
    
    test_root_endpoint()
    test_health_check()
    test_model_info()
    test_single_prediction()
    test_batch_prediction()
    
    print("\n" + "=" * 50)
    print("🏁 API Tests Completed!")

if __name__ == "__main__":
    main() 