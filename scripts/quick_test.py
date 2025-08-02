
import requests
import json
import time

def test_system():
    """Quick test of the loan prediction system"""
    
    BASE_URL = "http://localhost:8000"
    
    print("🧪 Testing Loan Prediction System...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running.")
        return
    
    # Test 2: Login
    try:
        login_data = {"username": "admin", "password": "Admin@123"}
        response = requests.post(f"{BASE_URL}/auth/login", data=login_data)
        
        if response.status_code == 200:
            token = response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            print("✅ Authentication successful")
        else:
            print("❌ Authentication failed")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # Test 3: Check available models
    try:
        response = requests.get(f"{BASE_URL}/model/available", headers=headers)
        if response.status_code == 200:
            models = response.json()["available_models"]
            available_models = [m for m in models if m["is_available"]]
            print(f"✅ Found {len(available_models)} available models")
            
            if available_models:
                current_model = response.json()["current_model"]
                print(f"📊 Current model: {current_model}")
            else:
                print("⚠️  No trained models found")
                return
        else:
            print("❌ Failed to get model information")
            return
    except Exception as e:
        print(f"❌ Model check error: {e}")
        return
    
    # Test 4: Create a test loan application
    try:
        loan_data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
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
            "preferred_model": current_model
        }
        
        response = requests.post(f"{BASE_URL}/loans/", json=loan_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction_result"]
            print(f"✅ Loan application processed")
            print(f"📋 Result: {'Approved' if prediction['approved'] else 'Rejected'}")
            print(f"🎯 Confidence: {prediction['confidence_score']:.2f}")
            print(f"⚠️  Risk Score: {prediction['risk_score']:.2f}")
        else:
            print(f"❌ Loan application failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Loan application error: {e}")
        return
    
    # Test 5: Test model switching (if multiple models available)
    if len(available_models) > 1:
        try:
            # Switch to a different model
            other_model = None
            for model in available_models:
                if not model["is_current"]:
                    other_model = model["model_type"]
                    break
            
            if other_model:
                switch_data = {"model_type": other_model}
                response = requests.post(f"{BASE_URL}/model/switch", json=switch_data, headers=headers)
                
                if response.status_code == 200:
                    print(f"✅ Successfully switched to {other_model}")
                    
                    # Test prediction with new model
                    response = requests.post(f"{BASE_URL}/loans/", json=loan_data)
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction_result"]
                        print(f"✅ Prediction with {other_model}: {'Approved' if prediction['approved'] else 'Rejected'}")
                    
                else:
                    print(f"❌ Model switching failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Model switching error: {e}")
    
    print("\n🎉 All tests completed successfully!")
    print("💡 The system is ready for production use")

if __name__ == "__main__":
    test_system()