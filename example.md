USAGE_EXAMPLES = """
# 🚀 LOAN PREDICTION SYSTEM - COMPLETE USAGE GUIDE

## 1. SYSTEM SETUP

### Quick Setup (Recommended)
```bash
# Run the complete setup script
python scripts/setup_system.py
```

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python scripts/generate_loan_data.py

# 3. Train models with hyperparameter tuning
python scripts/train_models.py --hyperparameter-tuning

# 4. Setup database
alembic upgrade head

# 5. Initialize system data
python scripts/init_data.py

# 6. Start the server
uvicorn app.main:app --reload
```

## 2. TRAINING MODELS

### Train All Models
```bash
python scripts/train_models.py --hyperparameter-tuning
```

### Train Specific Models
```bash
python scripts/train_models.py --models xgboost random_forest --hyperparameter-tuning
```

### Train with Custom Data
```bash
python scripts/train_models.py --data-file custom_data.csv --models xgboost
```

### Generate More Training Data
```bash
python scripts/generate_loan_data.py
# This will create 2500 records by default
# Edit the script to change the number
```

## 3. API USAGE EXAMPLES

### Authentication
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=Admin@123"
```

### Get Available Models
```bash
curl -X GET "http://localhost:8000/model/available" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Switch Model
```bash
curl -X POST "http://localhost:8000/model/switch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"model_type": "xgboost"}'
```

### Create Loan Application
```bash
curl -X POST "http://localhost:8000/loans/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "gender": "Male",
    "married": "Yes",
    "dependents": 2,
    "education": "Graduate",
    "age": 35,
    "applicant_income": 75000,
    "coapplicant_income": 25000,
    "monthly_expenses": 15000,
    "loan_amount": 300000,
    "loan_amount_term": 360,
    "credit_score": 720,
    "credit_history": 1,
    "property_area": "Urban",
    "preferred_model": "xgboost"
  }'
```

### Compare Model Performance
```bash
curl -X GET "http://localhost:8000/model/comparison" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 4. PYTHON CLIENT EXAMPLE

```python
import requests

class LoanPredictionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
    
    def login(self, username="admin", password="Admin@123"):
        response = requests.post(f"{self.base_url}/auth/login", 
                               data={"username": username, "password": password})
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            return True
        return False
    
    def get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}
    
    def get_available_models(self):
        response = requests.get(f"{self.base_url}/model/available", 
                              headers=self.get_headers())
        return response.json()
    
    def switch_model(self, model_type):
        response = requests.post(f"{self.base_url}/model/switch",
                               json={"model_type": model_type},
                               headers=self.get_headers())
        return response.json()
    
    def predict_loan(self, loan_data, preferred_model=None):
        if preferred_model:
            loan_data["preferred_model"] = preferred_model
        
        response = requests.post(f"{self.base_url}/loans/",
                               json=loan_data)
        return response.json()

# Usage example
client = LoanPredictionClient()
client.login()

# Get models
models = client.get_available_models()
print("Available models:", [m['model_type'] for m in models['available_models']])

# Switch to XGBoost
client.switch_model("xgboost")

# Make prediction
loan_data = {
    "name": "Jane Smith",
    "email": "jane@example.com",
    "gender": "Female",
    "married": "No",
    "dependents": 0,
    "education": "Graduate",
    "age": 28,
    "applicant_income": 60000,
    "loan_amount": 200000,
    "loan_amount_term": 240,
    "credit_score": 750,
    "credit_history": 1,
    "property_area": "Urban"
}

result = client.predict_loan(loan_data, preferred_model="random_forest")
print("Prediction:", result)
```

## 5. TESTING THE SYSTEM

### Quick Test
```bash
python scripts/quick_test.py
```

This will:
- Test API connectivity
- Authenticate with admin credentials
- Check available models
- Create a test loan application
- Test model switching
- Verify predictions work

## 6. PRODUCTION DEPLOYMENT

### Docker Deployment
```bash
# Build and run with docker-compose
docker-compose -f docker-compose.prod.yml up -d --build
```

### Manual Production Setup
```bash
# 1. Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/loan_db"
export SECRET_KEY="your-production-secret-key"

# 2. Run with production settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 7. MONITORING AND MAINTENANCE

### Check Model Performance
```bash
curl -X GET "http://localhost:8000/model/performance" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### View System Health
```bash
curl -X GET "http://localhost:8000/health/detailed"
```

### Generate New Training Data
```bash
# Generate more data for retraining
python scripts/generate_loan_data.py

# Retrain models with new data
python scripts/train_models.py --data-file data/loan_training_dataset.csv --hyperparameter-tuning
```
"""

if __name__ == "__main__":
    print(USAGE_EXAMPLES)