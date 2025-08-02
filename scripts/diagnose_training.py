# scripts/diagnose_training.py - Diagnostic script to identify training issues
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Mapping from pip package names to Python import names
PACKAGE_IMPORT_MAPPING = {
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "Pillow": "PIL",
    "python-dateutil": "dateutil",
    "opencv-python": "cv2",
     'pandas' : "pandas",
       "numpy": "numpy",
       "joblib" : "joblib",
         "faker" : "faker"
    # Add more mappings here if needed
}

def check_system_requirements():
    """Check if all system requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'joblib', 'faker'
    ]
    
    missing_packages = []
    for package in required_packages:
        import_name = PACKAGE_IMPORT_MAPPING.get(package, package)
        try:
            __import__(import_name)
            print(f"   ✅ {package}: Available")
        except ImportError:
            print(f"   ❌ {package}: Missing")
            missing_packages.append(package)
    
    # Check optional packages
    optional_packages = ['xgboost']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: Available")
        except ImportError:
            print(f"   ⚠️  {package}: Missing (optional)")
    
    return missing_packages

def check_data_files():
    """Check if data files exist and are valid"""
    print("\n📊 Checking data files...")
    
    data_files = [
        'data/loan_training_dataset.csv',
        # 'data/loan_training_dataset_2500.csv'
    ]
    
    valid_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file_path}: {len(df)} records, {len(df.columns)} columns")
                
                # Check required columns
                required_columns = ['loan_status', 'applicant_income', 'loan_amount']
                missing_cols = [col for col in required_columns if col not in df.columns]
                
                if missing_cols:
                    print(f"   ❌ Missing required columns: {missing_cols}")
                else:
                    print(f"   ✅ Required columns present")
                    valid_file = file_path
                    
                # Check data quality
                print(f"   📈 Approval rate: {(df['loan_status'] == 'Y').mean():.1%}")
                print(f"   🔍 Missing values: {df.isnull().sum().sum()}")
                
            except Exception as e:
                print(f"   ❌ {file_path}: Error reading file - {e}")
        else:
            print(f"   ❌ {file_path}: Not found")
    
    return valid_file

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = ['data', 'models', 'logs', 'scripts']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}: Exists")
        else:
            print(f"   ❌ {dir_path}: Missing")
            Path(dir_path).mkdir(exist_ok=True)
            print(f"   ✅ {dir_path}: Created")

def test_training_directly():
    """Test training script directly with detailed output"""
    print("\n🤖 Testing training script directly...")
    
    try:
        # Import training modules
        sys.path.append('.')
        sys.path.append('scripts')
        
        from scripts.train_models import ModelTrainer
        
        # Check if data file exists
        data_file = check_data_files()
        if not data_file:
            print("   ❌ No valid data file found for training")
            return False
        
        # Load data
        print(f"   📊 Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"   ✅ Data loaded: {len(df)} records")
        
        # Initialize trainer
        print("   🔧 Initializing trainer...")
        trainer = ModelTrainer(['xgboost'])  # Test with just one model first
        
        # Prepare data
        print("   🔄 Preparing data...")
        X, y, feature_names = trainer.prepare_data(df)
        print(f"   ✅ Data prepared: {X.shape}")
        
        # Train one model
        print("   🤖 Training XGBoost model...")
        results = trainer.train_models(X, y)
        
        if 'xgboost' in results and 'error' not in results['xgboost']:
            print("   ✅ Training successful!")
            
            # Save model
            trainer.save_models()
            print("   ✅ Models saved!")
            
            return True
        else:
            print(f"   ❌ Training failed: {results.get('xgboost', {}).get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if trained models can be loaded"""
    print("\n🔄 Testing model loading...")
    
    model_files = [
        'models/xgboost_model.joblib',
        'models/random_forest_model.joblib',
        'models/gradient_boosting_model.joblib',
        'models/logistic_regression_model.joblib'
    ]
    
    preprocessing_file = 'models/preprocessing_components.joblib'
    
    # Check preprocessing components
    if os.path.exists(preprocessing_file):
        try:
            import joblib
            components = joblib.load(preprocessing_file)
            print(f"   ✅ Preprocessing components loaded")
            print(f"   📋 Feature count: {len(components.get('feature_names', []))}")
        except Exception as e:
            print(f"   ❌ Error loading preprocessing: {e}")
    else:
        print(f"   ❌ Preprocessing file not found: {preprocessing_file}")
    
    # Check model files
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                import joblib
                model = joblib.load(model_file)
                model_name = os.path.basename(model_file).replace('_model.joblib', '')
                print(f"   ✅ {model_name}: Loaded successfully")
            except Exception as e:
                print(f"   ❌ {model_file}: Error loading - {e}")
        else:
            print(f"   ❌ {model_file}: Not found")

def generate_test_data():
    """Generate test data if needed"""
    print("\n📊 Generating test data...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "scripts/generate_loan_data.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Test data generated successfully")
            print("   📋 Output:")
            for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    print(f"      {line}")
            return True
        else:
            print(f"   ❌ Data generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error generating data: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🔍 LOAN PREDICTION SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    # Check system requirements
    missing_packages = check_system_requirements()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return
    
    # Check directories
    check_directories()
    
    # Check or generate data
    data_file = check_data_files()
    if not data_file:
        print("\n📊 No data file found, generating...")
        if generate_test_data():
            data_file = check_data_files()
    
    if not data_file:
        print("❌ Could not create or find valid data file")
        return
    
    # Test training
    if test_training_directly():
        print("\n✅ Training test successful!")
        test_model_loading()
    else:
        print("\n❌ Training test failed!")
        
        # Try simpler approach
        print("\n🔄 Trying manual training steps...")
        try:
            import subprocess
            
            # Try training with just one model
            cmd = [sys.executable, "scripts/train_models.py", 
                   "--models", "logistic_regression", 
                   "--data-file", data_file]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print("   STDOUT:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"      {line}")
            
            if result.stderr:
                print("   STDERR:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"      {line}")
            
        except Exception as e:
            print(f"   ❌ Manual training failed: {e}")
    
    print("\n🎯 DIAGNOSTIC COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()