from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/loan_db"
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # ML Model settings
    ml_model_path: str = "models/loan_model.joblib"
    ml_scaler_path: str = "models/scaler.joblib"
    default_model_type: str = "xgboost"

  # Additional fields with proper types
    app_name: str = "Loan Prediction System"
    app_version: str = "1.0.0"
    debug: bool = False

    # Optional email settings
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra environment variables
        protected_namespaces = ('settings_',)  # Resolve namespace conflicts

settings = Settings()