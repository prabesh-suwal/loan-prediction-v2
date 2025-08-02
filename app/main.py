from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .config.database import engine, Base
from .api import loans, weights, auth, users, health

from .api import model_management, reports
from .core.exceptions import (
    LoanApplicationError, MLPredictionError, WeightConfigurationError,
    http_exception_handler, validation_exception_handler,
    loan_application_exception_handler, ml_prediction_exception_handler,
    weight_configuration_exception_handler, general_exception_handler
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .core.dependencies import get_current_user  # Add this import
from app.utils.logging_config import setup_logging
from app.middleware.performance import PerformanceMiddleware
import os


DEBUG = os.getenv("DEBUG", "False").lower() == "true"
logger = setup_logging()

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Loan Prediction System",
    description="AI/ML-powered loan approval prediction system",
    version="1.0.0",
    debug=DEBUG  # Add this line
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(loans.router)
app.include_router(weights.router)
app.include_router(model_management.router)
app.include_router(reports.router)
app.include_router(health.router)

# Add exception handlers to main.py
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(LoanApplicationError, loan_application_exception_handler)
app.add_exception_handler(MLPredictionError, ml_prediction_exception_handler)
app.add_exception_handler(WeightConfigurationError, weight_configuration_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)



from app.api import health
app.add_middleware(PerformanceMiddleware)

@app.get("/")
async def root():
    return {
        "message": "Loan Prediction System API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
