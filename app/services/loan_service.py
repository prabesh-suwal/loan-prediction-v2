from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from ..repositories.loan_repository import LoanRepository
from ..repositories.weight_repository import WeightRepository
from ..schemas.loan_application import LoanApplicationCreate, PredictionResult
from ..models.loan_application import LoanApplication
from .ml_service import MLService

class LoanService:
    def __init__(self, db: Session):
        self.db = db
        self.loan_repository = LoanRepository(db)
        weight_repository = WeightRepository(db)
        self.ml_service = MLService(weight_repository)
    
    def create_loan_application(self, loan_data: LoanApplicationCreate, model_type: str = None) -> LoanApplication:
        """Create new loan application with ML prediction using specified model"""
        
        # Switch model if specified
        if model_type and model_type != self.ml_service.current_model_type:
            try:
                self.ml_service.switch_model(model_type)
            except Exception as e:
                # Log warning but continue with current model
                print(f"Warning: Could not switch to {model_type}, using {self.ml_service.current_model_type}: {e}")
        
        # Convert Pydantic model to dict
        loan_dict = loan_data.dict()
        
        # Get ML prediction
        prediction_result = self.ml_service.predict_loan_approval(loan_dict)
        
        # Add prediction results to loan data
        loan_dict.update({
            'is_approved': prediction_result['approved'],
            'confidence_score': prediction_result['confidence_score'],
            'risk_score': prediction_result['risk_score'],
            'prediction_details': prediction_result['prediction_details'],
            'recommended_interest_rate': prediction_result['recommended_interest_rate'],
            'conditions': prediction_result['conditions'],
            'status': 'processed',
            'model_used': prediction_result.get('model_used'),
            'model_performance': prediction_result.get('model_performance')
        })
        
        # Remove the key
        loan_dict.pop('preferred_model', None)
        # Save to database
        loan_application = self.loan_repository.create(loan_dict)
        
        return loan_application
    
    def get_loan_application(self, loan_id: int) -> Optional[LoanApplication]:
        """Get loan application by ID"""
        return self.loan_repository.get_by_id(loan_id)
    
    def get_loan_applications(self, 
                            skip: int = 0, 
                            limit: int = 100,
                            **filters) -> List[LoanApplication]:
        """Get paginated list of loan applications with filters"""
        return self.loan_repository.search_applications(skip=skip, limit=limit, **filters)
    
    def update_loan_status(self, loan_id: int, status: str) -> Optional[LoanApplication]:
        """Update loan application status"""
        return self.loan_repository.update(loan_id, {'status': status})
    
    def get_model_usage_stats(self) -> Dict[str, Any]:
        """Get statistics on model usage"""
        return {
            'current_model': self.ml_service.current_model_type,
            'available_models': self.ml_service.get_available_models(),
            'model_comparison': self.ml_service.get_model_comparison()
        }