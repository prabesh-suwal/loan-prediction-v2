from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    role: str = Field(default="RM", pattern="^(superadmin|RM)$")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, pattern="^(superadmin|RM)$")
    password: Optional[str] = Field(None, min_length=8)
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    role: str
    expires_in: int

# app/schemas/field_weight.py
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