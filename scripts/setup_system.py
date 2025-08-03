import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}... with command: {command}")
    try:

        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():  # Only print stdout if there's meaningful content
            # Print last few lines of output for feedback
            lines = result.stdout.strip().split('\n')
            if len(lines) > 5:
                print("   Output (last 5 lines):")
                for line in lines[-5:]:
                    if line.strip():
                        print(f"   {line}")
            else:
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        return None

def main():
    """Setup complete loan prediction system"""
    print("🚀 Setting up Complete Loan Prediction System...")
    
    # Create necessary directories
    directories = ['data', 'models', 'logs', 'scripts']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install requirements
    print("📦 Installing Python dependencies...")
    if run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("✅ Dependencies installed successfully")
    else:
        print("⚠️  Some dependencies may have failed to install")
    
    # Generate sample data
    print("📊 Generating sample loan dataset...")
    if run_command("python scripts/generate_loan_data.py", "Generating sample data"):
        print("✅ Sample data generated successfully")
    else:
        print("❌ Failed to generate sample data")
        return
    
    # Train models
    print("🤖 Training ML models...")
    # if run_command("python scripts/train_models.py --hyperparameter-tuning", "Training ML models"):
    if run_command("python scripts/train_models.py", "Training ML models"):

        print("✅ Models trained successfully")
    else:
        print("❌ Failed to train models")
        return
    
    # Setup database
    print("🗄️  Setting up database...")
    run_command("alembic upgrade head", "Setting up database")
    print("✅ Database setup step completed")
    
    print("\n🎉 System setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the application: uvicorn app.main:app --reload")
    print("2. Open browser: http://localhost:8000/docs")
    print("3. Login with admin credentials (admin / Admin@123)")
    print("4. Test loan applications with different models")
    print("5. Switch between models: POST /model/switch")
    print("6. Compare model performance: GET /model/comparison")

if __name__ == "__main__":
    main()
