from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict
from ..config.database import get_db
from ..services.weight_service import WeightService
from ..core.dependencies import get_current_superadmin
from pydantic import BaseModel
from ..schemas.field_weight import FieldWeightBulkUpdate

router = APIRouter(prefix="/weights", tags=["weights"])

class WeightUpdate(BaseModel):
    field_name: str
    weight: float

class WeightResponse(BaseModel):
    id: int
    field_name: str
    weight: float
    is_active: bool
    description: str = None

@router.get("/", response_model=List[WeightResponse])
async def get_all_weights(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Get all field weights (superadmin only)"""
    
    weight_service = WeightService(db)
    weights = weight_service.get_all_weights()
    
    return [
        WeightResponse(
            id=weight.id,
            field_name=weight.field_name,
            weight=weight.weight,
            is_active=weight.is_active,
            description=weight.description
        )
        for weight in weights
    ]

@router.put("/")
async def update_field_weight(
    weight_update: WeightUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Update field weight (superadmin only)"""
    
    weight_service = WeightService(db)
    
    try:
        updated_weight = weight_service.update_field_weight(
            weight_update.field_name, 
            weight_update.weight
        )
        
        if not updated_weight:
            raise HTTPException(status_code=404, detail="Field weight not found")
        
        return {
            "message": f"Weight for {weight_update.field_name} updated successfully",
            "field_name": weight_update.field_name,
            "new_weight": weight_update.weight
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating weight: {str(e)}")

@router.post("/initialize")
async def initialize_default_weights(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Initialize default field weights (superadmin only)"""
    
    weight_service = WeightService(db)
    
    try:
        weight_service.initialize_default_weights()
        return {"message": "Default weights initialized successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing weights: {str(e)}")
    
@router.put("/bulk")
async def bulk_update_weights(
    weights_data: FieldWeightBulkUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Bulk update field weights (superadmin only)"""
    
    weight_service = WeightService(db)
    
    try:
        updated_weights = []
        errors = []
        
        for field_name, weight in weights_data.weights.items():
            try:
                updated_weight = weight_service.update_field_weight(field_name, weight)
                if updated_weight:
                    updated_weights.append({
                        "field_name": field_name,
                        "weight": weight,
                        "status": "updated"
                    })
                else:
                    errors.append({
                        "field_name": field_name,
                        "error": "Field not found or could not be updated"
                    })
            except Exception as e:
                errors.append({
                    "field_name": field_name,
                    "error": str(e)
                })
        
        return {
            "message": f"Bulk update completed. {len(updated_weights)} weights updated.",
            "updated_weights": updated_weights,
            "errors": errors
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in bulk update: {str(e)}")

