from typing import Dict, Any, List, Optional
from ..ml.predictor import LoanPredictor
from ..repositories.weight_repository import WeightRepository
from ..config.settings import settings
import joblib
import json
import os

class MLService:
    def __init__(self, weight_repository: WeightRepository):
        self.weight_repository = weight_repository
        self.available_models = ['xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression']
        self.current_model_type = 'xgboost'  # Default model
        self.predictor = None
        self.model_performances = {}
        self._load_model_performances()
        self._initialize_predictor()
    
    def _load_model_performances(self):
        """Load model performance metrics"""
        try:
            performance_path = "models/model_performances.json"
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.model_performances = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model performances: {e}")
            self.model_performances = {}
    
    def _initialize_predictor(self):
        """Initialize predictor with current model"""
        try:
            model_path = f"models/{self.current_model_type}_model.joblib"
            preprocessing_path = "models/preprocessing_components.joblib"
            
            if os.path.exists(model_path) and os.path.exists(preprocessing_path):
                self.predictor = LoanPredictor(model_path, preprocessing_path)
            else:
                print(f"Warning: Model files not found for {self.current_model_type}")
                self.predictor = None
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            self.predictor = None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available trained models with their performance"""
        models_info = []
        
        for model_type in self.available_models:
            model_path = f"models/{model_type}_model.joblib"
            is_available = os.path.exists(model_path)
            
            performance = self.model_performances.get(model_type, {})
            
            models_info.append({
                'model_type': model_type,
                'is_available': is_available,
                'is_current': model_type == self.current_model_type,
                'performance': {
                    'accuracy': performance.get('accuracy', 0),
                    'roc_auc': performance.get('roc_auc', 0),
                    'cv_mean': performance.get('cv_mean', 0)
                }
            })
        
        return models_info
    
    def switch_model(self, model_type: str) -> Dict[str, Any]:
        """Switch to a different model type"""
        if model_type not in self.available_models:
            raise ValueError(f"Model type {model_type} not supported. Available: {self.available_models}")
        
        model_path = f"models/{model_type}_model.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for {model_type}")
        
        old_model = self.current_model_type
        self.current_model_type = model_type
        self._initialize_predictor()
        
        if self.predictor is None:
            # Rollback if initialization failed
            self.current_model_type = old_model
            self._initialize_predictor()
            raise Exception(f"Failed to initialize {model_type} model")
        
        return {
            'message': f'Successfully switched from {old_model} to {model_type}',
            'current_model': model_type,
            'performance': self.model_performances.get(model_type, {})
        }
    
    def predict_loan_approval(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict loan approval with current model and weights"""
        if self.predictor is None:
            raise Exception("No model is currently loaded")
        
        # Get current field weights
        weights = self.weight_repository.get_active_weights_dict()
        
        # Make prediction
        prediction_result = self.predictor.predict(loan_data, weights)
        
        # Add model information to result
        prediction_result['model_used'] = self.current_model_type
        prediction_result['model_performance'] = self.model_performances.get(self.current_model_type, {})
        
        return prediction_result
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get comparison of all available models"""
        models_info = self.get_available_models()
        
        # Sort by ROC-AUC score
        available_models = [m for m in models_info if m['is_available']]
        available_models.sort(key=lambda x: x['performance']['roc_auc'], reverse=True)
        
        return {
            'total_models': len(self.available_models),
            'available_models': len(available_models),
            'current_model': self.current_model_type,
            'best_model': available_models[0]['model_type'] if available_models else None,
            'models': models_info
        }