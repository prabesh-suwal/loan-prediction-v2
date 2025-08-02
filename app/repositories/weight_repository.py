from typing import Optional, Dict
from sqlalchemy.orm import Session
from .base_repository import BaseRepository
from ..models.field_weight import FieldWeight

class WeightRepository(BaseRepository[FieldWeight]):
    def __init__(self, db: Session):
        super().__init__(db, FieldWeight)
    
    def get_by_field_name(self, field_name: str) -> Optional[FieldWeight]:
        return self.db.query(FieldWeight).filter(FieldWeight.field_name == field_name).first()
    
    def get_active_weights_dict(self) -> Dict[str, float]:
        """Get active weights as dictionary"""
        weights = self.db.query(FieldWeight).filter(FieldWeight.is_active == True).all()
        return {weight.field_name: weight.weight for weight in weights}