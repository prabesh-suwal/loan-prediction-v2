# scripts/simple_train.py - Simplified training script that should work
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
import json

def simple_train():
    """Simple training script that handles common issues"""
    print("🚀 Starting Simple Training...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Find data file
    data_files = [
        'data/loan_training_dataset.csv',
        'data/loan_training_dataset_2500.csv'
    ]
    
    data_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_file = file_path
            break
    
    if not data_file:
        print("❌ No data file found. Generate data first:")
        print("   python scripts/generate_loan_data.py")
        return False
    
    print(f"📊 Loading data from: {data_file}")
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded: {len(df)} records, {len(df.columns)} columns")
        
        # Check required columns
        if 'loan_status' not in df.columns:
            print("❌ Missing 'loan_status' column")
            return False
        
        # Prepare features and target
        X = df.drop(columns=['loan_status'])
        y = (df['loan_status'] == 'Y').astype(int)
        
        # Remove string columns that cause issues
        string_columns = ['name', 'email']
        for col in string_columns:
            if col in X.columns:
                X = X.drop(columns=[col])
                print(f"   Removed string column: {col}")
        
        print(f"📋 Features: {list(X.columns)}")
        print(f"📊 Target distribution: {np.bincount(y)} (0: Reject, 1: Approve)")
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'bool']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"   Encoded categorical: {col}")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ Data prepared for training")
        
        # Train models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        model_performances = {}
        
        for model_name, model in models.items():
            print(f"🤖 Training {model_name}...")
            
            try:
                # Train model
                if model_name == 'logistic_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                y_pred = (y_pred_proba > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                print(f"   ✅ {model_name}: Accuracy={accuracy:.3f}, ROC-AUC={roc_auc:.3f}")
                
                # Save model
                model_path = f"models/{model_name}_model.joblib"
                joblib.dump(model, model_path)
                
                model_performances[model_name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'cv_mean': roc_auc  # Simplified
                }
                
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
        
        # Save preprocessing components
        preprocessing_components = {
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': list(X.columns)
        }
        
        preprocessing_path = "models/preprocessing_components.joblib"
        joblib.dump(preprocessing_components, preprocessing_path)
        print(f"💾 Saved preprocessing components")
        
        # Save model performances
        performance_path = "models/model_performances.json"
        with open(performance_path, 'w') as f:
            json.dump(model_performances, f, indent=2)
        print(f"💾 Saved model performances")
        
        print(f"🎉 Training completed successfully!")
        print(f"📊 Trained {len(model_performances)} models")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction():
    """Test if prediction works with trained models"""
    print("\n🧪 Testing prediction...")
    
    try:
        # Test data
        test_data = {
            "gender": "Male",
            "married": "Yes",
            "dependents": 1,
            "education": "Graduate",
            "age": 32,
            "self_employed": "No",
            "employment_type": "Salaried",
            "years_in_current_job": 2,
            "employer_category": "B",
            "industry": "IT",
            "spouse_employed": False,
            "applicant_income": 75000,
            "coapplicant_income": 25000,
            "monthly_expenses": 15000,
            "other_emis": 0,
            "loan_amount": 300000,
            "loan_amount_term": 360,
            "loan_purpose": "Home",
            "credit_score": 720,
            "credit_history": 1,
            "no_of_credit_cards": 2,
            "loan_default_history": 0,
            "avg_payment_delay_days": 0,
            "has_vehicle": True,
            "has_life_insurance": True,
            "property_area": "Urban",
            "bank_account_type": "Savings",
            "bank_balance": 100000,
            "savings_score": 15,
            "collateral_type": "Property",
            "collateral_value": 500000,
            "city_tier": "Tier-1",
            "pincode": "110001",
            "region_default_rate": 3.5
        }
        
        # Load components
        preprocessing_path = "models/preprocessing_components.joblib"
        model_path = "models/logistic_regression_model.joblib"
        
        if not os.path.exists(preprocessing_path) or not os.path.exists(model_path):
            print("❌ Model files not found")
            return False
        
        # Load model and preprocessor
        preprocessing_components = joblib.load(preprocessing_path)
        model = joblib.load(model_path)
        
        # Prepare test data
        df = pd.DataFrame([test_data])
        
        # Encode categorical variables
        for col, encoder in preprocessing_components['label_encoders'].items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0  # Fallback for unseen categories
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Scale features
        X_scaled = preprocessing_components['scaler'].transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        print(f"✅ Prediction test successful!")
        print(f"   Result: {'Approved' if prediction else 'Rejected'}")
        print(f"   Confidence: {max(prediction_proba):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if simple_train():
        test_prediction()
    else:
        print("\n💡 Try running diagnostic script:")
        print("   python scripts/diagnose_training.py")