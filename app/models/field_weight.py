from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from ..config.database import Base

class FieldWeight(Base):
    __tablename__ = "field_weights"
    
    id = Column(Integer, primary_key=True, index=True)
    field_name = Column(String(100), unique=True, nullable=False)
    weight = Column(Float, default=1.0)
    is_active = Column(Boolean, default=True)
    description = Column(String(255), nullable=True)
    category = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())