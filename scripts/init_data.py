import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.orm import Session
from app.config.database import SessionLocal, engine, Base
from app.services.user_service import UserService
from app.services.weight_service import WeightService
from app.core.security import get_password_hash
from app.models.user import User

def create_initial_superadmin():
    # \"\"\"Create initial superadmin user\"\"\"
    db = SessionLocal()
    
    try:
        # Check if superadmin already exists
        existing_admin = db.query(User).filter(User.role == "superadmin").first()
        
        if not existing_admin:
            # Create superadmin
            superadmin = User(
                username="admin",
                email="admin@loanpredict.com",
                hashed_password=get_password_hash("Admin@123"),
                full_name="System Administrator",
                role="superadmin",
                is_active=True
            )
            
            db.add(superadmin)
            db.commit()
            print("✅ Initial superadmin created successfully")
            print("Username: admin")
            print("Password: Admin@123")
            print("⚠️  Please change the default password after first login")
        else:
            print("✅ Superadmin already exists")
    
    except Exception as e:
        print(f"❌ Error creating superadmin: {e}")
        db.rollback()
    
    finally:
        db.close()

def initialize_field_weights():
    # \"\"\"Initialize default field weights\"\"\"
    db = SessionLocal()
    
    try:
        weight_service = WeightService(db)
        weight_service.initialize_default_weights()
        print("✅ Default field weights initialized")
    
    except Exception as e:
        print(f"❌ Error initializing weights: {e}")
    
    finally:
        db.close()

def create_sample_rm_user():
    # \"\"\"Create a sample RM user for testing\"\"\"
    db = SessionLocal()
    
    try:
        # Check if RM user already exists
        existing_rm = db.query(User).filter(User.username == "rm_user").first()
        
        if not existing_rm:
            rm_user = User(
                username="rm_user",
                email="rm@loanpredict.com",
                hashed_password=get_password_hash("RM@123"),
                full_name="Relationship Manager",
                role="RM",
                is_active=True
            )
            
            db.add(rm_user)
            db.commit()
            print("✅ Sample RM user created successfully")
            print("Username: rm_user")
            print("Password: RM@123")
        else:
            print("✅ RM user already exists")
    
    except Exception as e:
        print(f"❌ Error creating RM user: {e}")
        db.rollback()
    
    finally:
        db.close()

def main():
    # \"\"\"Main initialization function\"\"\"
    print("🚀 Initializing Loan Prediction System...")
    
    # Create database tables
    print("📊 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
    
    # Create initial users
    print("👤 Creating initial users...")
    create_initial_superadmin()
    create_sample_rm_user()
    
    # Initialize field weights
    print("⚖️ Initializing field weights...")
    initialize_field_weights()
    
    print("🎉 System initialization completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the application: uvicorn app.main:app --reload")
    print("2. Login with admin credentials")
    print("3. Change default passwords")
    print("4. Configure field weights as needed")
    print("5. Train your ML model with historical data")

if __name__ == "__main__":
    main()