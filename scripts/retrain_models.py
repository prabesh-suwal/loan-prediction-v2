import pandas as pd
import numpy as np
import os
import subprocess
import sys

def retrain_models_with_feature_alignment():
    """Retrain models to ensure feature alignment between training and prediction"""
    
    print("🔄 Retraining models to fix feature alignment...")
    
    # Check if data exists
    data_file = "data/loan_training_dataset.csv"
    if not os.path.exists(data_file):
        print("📊 Generating new training data...")
        subprocess.run([sys.executable, "scripts/generate_loan_data.py"], check=True)
    
    # Retrain models
    print("🤖 Training models with correct feature engineering...")
    cmd = [
        sys.executable, "scripts/train_models.py",
        "--data-file", data_file,
        "--hyperparameter-tuning",
        "--output-dir", "models/"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Models retrained successfully!")
        print("📋 Training output:")
        print(result.stdout)
    else:
        print("❌ Training failed:")
        print(result.stderr)
        return False
    
    # Test prediction
    print("🧪 Testing prediction with retrained models...")
    try:
        test_prediction()
        print("✅ Prediction test passed!")
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_prediction():
    """Test prediction with sample data"""
    import sys
    sys.path.append('.')
    
    from app.ml.predictor import LoanPredictor
    
    # Sample data matching your curl request
    test_data = {
        "name": "Test User",
        "email": "test@example.com",
        "gender": "Male",
        "married": "Yes",
        "dependents": 1,
        "education": "Graduate",
        "age": 32,
        "children": 0,
        "self_employed": "Yes",
        "employment_type": "Salaried",
        "years_in_current_job": 2,
        "employer_category": "B",
        "industry": "Others",
        "spouse_employed": False,
        "applicant_income": 2112312,
        "coapplicant_income": 0,
        "monthly_expenses": 1232,
        "other_emis": 0,
        "loan_amount": 123122,
        "loan_amount_term": 12,
        "loan_purpose": "Personal",
        "credit_score": 650,
        "credit_history": 1,
        "no_of_credit_cards": 1,
        "loan_default_history": 0,
        "avg_payment_delay_days": 0,
        "has_vehicle": False,
        "has_life_insurance": False,
        "property_area": "Urban",
        "bank_account_type": "Savings",
        "bank_balance": 50000,
        "savings_score": 10,
        "collateral_type": "Property",
        "collateral_value": 0,
        "city_tier": "Tier-2",
        "pincode": "110001",
        "region_default_rate": 5
    }
    
    # Test with each available model
    models = ['xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression']
    
    for model_type in models:
        model_path = f"models/{model_type}_model.joblib"
        preprocessing_path = "models/preprocessing_components.joblib"
        
        if os.path.exists(model_path) and os.path.exists(preprocessing_path):
            print(f"Testing {model_type}...")
            predictor = LoanPredictor(model_path, preprocessing_path)
            result = predictor.predict(test_data)
            print(f"  ✅ {model_type}: {'Approved' if result['approved'] else 'Rejected'} (confidence: {result['confidence_score']:.2f})")

if __name__ == "__main__":
    retrain_models_with_feature_alignment()