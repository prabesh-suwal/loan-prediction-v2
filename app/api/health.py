from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config.database import get_db
from app.ml.predictor import LoanPredictor
from app.config.settings import settings
import os
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    # \"\"\"Basic health check\"\"\"
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Loan Prediction System",
        "version": "1.0.0"
    }

@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    # \"\"\"Detailed health check including dependencies\"\"\"
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database connectivity check
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {"status": "healthy", "message": "Connected"}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "message": str(e)}
        health_status["status"] = "unhealthy"
    
    # ML model check
    try:
        if os.path.exists(settings.ml_model_path):
            predictor = LoanPredictor(settings.ml_model_path, settings.ml_scaler_path)
            if predictor.model is not None:
                health_status["checks"]["ml_model"] = {"status": "healthy", "message": "Model loaded"}
            else:
                health_status["checks"]["ml_model"] = {"status": "unhealthy", "message": "Model not loaded"}
                health_status["status"] = "degraded"
        else:
            health_status["checks"]["ml_model"] = {"status": "unhealthy", "message": "Model file not found"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["ml_model"] = {"status": "unhealthy", "message": str(e)}
        health_status["status"] = "degraded"
    
    # Disk space check
    try:
        disk_usage = os.statvfs('.')
        free_space_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
        
        if free_space_gb > 1:  # More than 1GB free
            health_status["checks"]["disk_space"] = {
                "status": "healthy", 
                "message": f"{free_space_gb:.2f}GB free"
            }
        else:
            health_status["checks"]["disk_space"] = {
                "status": "warning", 
                "message": f"Low disk space: {free_space_gb:.2f}GB free"
            }
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["disk_space"] = {"status": "unknown", "message": str(e)}
    
    return health_status

@router.get("/readiness")
async def readiness_check(db: Session = Depends(get_db)):
    # \"\"\"Readiness probe for Kubernetes\"\"\"
    
    try:
        # Check database
        db.execute(text("SELECT 1"))
        
        # Check if ML model exists
        if not os.path.exists(settings.ml_model_path):
            raise HTTPException(status_code=503, detail="ML model not available")
        
        return {"status": "ready"}
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@router.get("/liveness")
async def liveness_check():
    # \"\"\"Liveness probe for Kubernetes\"\"\"
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}