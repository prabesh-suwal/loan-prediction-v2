from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)

class LoanApplicationError(Exception):
    """Custom exception for loan application errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class MLPredictionError(Exception):
    """Custom exception for ML prediction errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class WeightConfigurationError(Exception):
    """Custom exception for weight configuration errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation Error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
    )

async def loan_application_exception_handler(request: Request, exc: LoanApplicationError):
    """Handle loan application errors"""
    logger.error(f"Loan Application Error: {exc.message}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.error_code or "LOAN_APPLICATION_ERROR",
                "message": exc.message,
                "type": "loan_application_error"
            }
        }
    )

async def ml_prediction_exception_handler(request: Request, exc: MLPredictionError):
    """Handle ML prediction errors"""
    logger.error(f"ML Prediction Error: {exc.message}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": exc.error_code or "ML_PREDICTION_ERROR",
                "message": exc.message,
                "type": "ml_prediction_error"
            }
        }
    )

async def weight_configuration_exception_handler(request: Request, exc: WeightConfigurationError):
    """Handle weight configuration errors"""
    logger.error(f"Weight Configuration Error: {exc.message}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.error_code or "WEIGHT_CONFIG_ERROR",
                "message": exc.message,
                "type": "weight_configuration_error"
            }
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected Error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "type": "server_error"
            }
        }
    )
