from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from ..repositories.weight_repository import WeightRepository
from ..models.field_weight import FieldWeight

class WeightService:
    def __init__(self, db: Session):
        self.db = db
        self.weight_repository = WeightRepository(db)
    
    def initialize_default_weights(self):
        """Initialize default field weights"""
        
        default_weights = {
            # High importance factors
            'credit_score': 2.0,
            'credit_history': 2.0,
            'loan_default_history': 1.8,
            'applicant_income': 1.5,
            'loan_amount': 1.5,
            
            # Medium importance factors
            'employment_type': 1.2,
            'years_in_current_job': 1.2,
            'property_area': 1.1,
            'education': 1.1,
            'collateral_value': 1.3,
            
            # Lower importance factors
            'gender': 0.8,
            'married': 0.9,
            'dependents': 0.9,
            'age': 1.0,
            'has_vehicle': 0.8,
            'has_life_insurance': 0.9,
        }
        
        for field_name, weight in default_weights.items():
            existing_weight = self.weight_repository.get_by_field_name(field_name)
            if not existing_weight:
                self.weight_repository.create({
                    'field_name': field_name,
                    'weight': weight,
                    'is_active': True,
                    'description': f'Weight for {field_name} field'
                })
    
    def update_field_weight(self, field_name: str, weight: float) -> Optional[FieldWeight]:
        """Update field weight"""
        existing_weight = self.weight_repository.get_by_field_name(field_name)
        
        if existing_weight:
            return self.weight_repository.update(existing_weight.id, {'weight': weight})
        else:
            return self.weight_repository.create({
                'field_name': field_name,
                'weight': weight,
                'is_active': True
            })
    
    def get_all_weights(self) -> List[FieldWeight]:
        """Get all field weights"""
        return self.weight_repository.get_all()
    
    def get_active_weights(self) -> Dict[str, float]:
        """Get active field weights as dictionary"""
        return self.weight_repository.get_active_weights_dict()