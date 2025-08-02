from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FieldWeightBase(BaseModel):
    field_name: str = Field(..., min_length=1, max_length=100)
    weight: float = Field(..., ge=0.0, le=10.0)
    description: Optional[str] = Field(None, max_length=255)
    category: Optional[str] = Field(None, max_length=50)

class FieldWeightCreate(FieldWeightBase):
    is_active: bool = Field(default=True)

class FieldWeightUpdate(BaseModel):
    weight: Optional[float] = Field(None, ge=0.0, le=10.0)
    description: Optional[str] = Field(None, max_length=255)
    category: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None

class FieldWeightResponse(FieldWeightBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class FieldWeightBulkUpdate(BaseModel):
    weights: dict[str, float] = Field(..., description="Dictionary of field_name: weight pairs")