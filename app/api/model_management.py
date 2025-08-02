from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import pandas as pd
import io
import os
from ..config.database import get_db
from ..core.dependencies import get_current_superadmin
from ..services.ml_service import MLService
from ..repositories.weight_repository import WeightRepository
from ..config.settings import settings
from pydantic import BaseModel
import json

router = APIRouter(prefix="/model", tags=["model_management"])

class ModelSwitchRequest(BaseModel):
    model_type: str

class TrainingRequest(BaseModel):
    hyperparameter_tuning: bool = True
    models_to_train: List[str] = ['xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression']


@router.get("/available")
async def get_available_models(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Get list of available trained models"""
    
    weight_repository = WeightRepository(db)
    ml_service = MLService(weight_repository)
    
    try:
        models_info = ml_service.get_available_models()
        return {
            "available_models": models_info,
            "current_model": ml_service.current_model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")

@router.post("/switch")
async def switch_model(
    switch_request: ModelSwitchRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Switch to a different model type"""
    
    weight_repository = WeightRepository(db)
    ml_service = MLService(weight_repository)
    
    try:
        result = ml_service.switch_model(switch_request.model_type)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching model: {str(e)}")

@router.get("/comparison")
async def get_model_comparison(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Get comparison of all available models"""
    
    weight_repository = WeightRepository(db)
    ml_service = MLService(weight_repository)
    
    try:
        comparison = ml_service.get_model_comparison()
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model comparison: {str(e)}")

@router.post("/train")
async def train_models(
    training_request: TrainingRequest,
    training_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Train multiple ML models with uploaded data"""
    
    if not training_file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel format")
    
    try:
        # Read uploaded file
        contents = await training_file.read()
        
        if training_file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # Validate required columns
        required_columns = ['loan_status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Import and initialize enhanced trainer
        from ..ml.model_trainer import EnhancedModelTrainer
        
        trainer = EnhancedModelTrainer()
        
        # Prepare data
        X, y, feature_names = trainer.prepare_data(df)
        
        # Filter models to train
        if training_request.models_to_train:
            trainer.models = {k: v for k, v in trainer.models.items() 
                            if k in training_request.models_to_train}
        
        # Train models
        results = trainer.train_all_models(X, y)
        
        # Hyperparameter tuning if requested
        tuning_results = {}
        if training_request.hyperparameter_tuning:
            top_models = [name for name, result in results.items() 
                         if isinstance(result, dict) and 'error' not in result][:2]
            tuning_results = trainer.hyperparameter_tuning(X, y, top_models)
            
            # Re-train with tuned parameters
            results = trainer.train_all_models(X, y)
        
        # Save models
        trainer.save_models()
        
        # Generate comprehensive report
        report = trainer.generate_training_report(results)
        report['hyperparameter_tuning'] = tuning_results
        
        # Save training report
        import json
        with open('models/training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return {
            "message": "Model training completed successfully",
            "training_report": report,
            "models_trained": list(trainer.trained_models.keys()),
            "best_model": report['training_summary']['best_model'],
            "best_score": report['training_summary']['best_score']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@router.get("/performance")
async def get_model_performance(
    model_type: str = Query(None, description="Specific model type to get performance for"),
    current_user = Depends(get_current_superadmin)
):
    """Get model performance metrics"""
    
    try:
        # Load model performances
        performance_path = "models/model_performances.json"
        training_report_path = "models/training_report.json"
        
        performance_data = {}
        training_report = {}
        
        if os.path.exists(performance_path):
            with open(performance_path, 'r') as f:
                performance_data = json.load(f)
        
        if os.path.exists(training_report_path):
            with open(training_report_path, 'r') as f:
                training_report = json.load(f)
        
        if model_type:
            # Return specific model performance
            if model_type not in performance_data:
                raise HTTPException(status_code=404, detail=f"Performance data not found for {model_type}")
            
            return {
                "model_type": model_type,
                "performance": performance_data[model_type],
                "model_exists": os.path.exists(f"models/{model_type}_model.joblib")
            }
        else:
            # Return all model performances
            return {
                "all_performances": performance_data,
                "training_report": training_report,
                "available_models": [
                    model for model in performance_data.keys()
                    if os.path.exists(f"models/{model}_model.joblib")
                ]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

# @router.post("/generate-sample-data")
# async def generate_sample_data(
#     num_records: int = Query(2500, ge=100, le=10000),
#     current_user = Depends(get_current_superadmin)
# ):
#     """Generate sample loan data for training"""
    
#     try:
#         # Import the data generator
#         import sys
#         import os
        
#         # Add the scripts directory to Python path
#         script_dir = os.path.join(os.getcwd(), 'scripts')
#         if script_dir not in sys.path:
#             sys.path.append(script_dir)
        
#         from generate_loan_data import LoanDataGenerator
        
#         # Create data directory if it doesn't exist
#         os.makedirs('data', exist_ok=True)
        
#         # Generate dataset
#         generator = LoanDataGenerator(num_records=num_records)
#         df = generator.generate_realistic_dataset()
        
#         # Save dataset
#         filename = f'data/loan_training_dataset_{num_records}.csv'
#         generator.save_dataset(df, filename)
        
#         # Get summary statistics
#         summary = {
#             'total_records': len(df),
#             'approval_rate': (df['loan_status'] == 'Y').mean(),
#             'avg_loan_amount': df['loan_amount'].mean(),
#             'avg_applicant_income': df['applicant_income'].mean(),
#             'avg_credit_score': df['credit_score'].mean(),
#             'filename': filename
#         }
        
#         return {
#             "message": f"Successfully generated {num_records} loan records",
#             "filename": filename,
#             "summary": summary
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

@router.delete("/{model_type}")
async def delete_model(
    model_type: str,
    current_user = Depends(get_current_superadmin)
):
    """Delete a specific trained model"""
    
    try:
        model_path = f"models/{model_type}_model.joblib"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        # Remove model file
        os.remove(model_path)
        
        # Update performance file
        performance_path = "models/model_performances.json"
        if os.path.exists(performance_path):
            with open(performance_path, 'r') as f:
                performances = json.load(f)
            
            if model_type in performances:
                del performances[model_type]
                
                with open(performance_path, 'w') as f:
                    json.dump(performances, f, indent=2)
        
        return {
            "message": f"Successfully deleted {model_type} model",
            "deleted_model": model_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")