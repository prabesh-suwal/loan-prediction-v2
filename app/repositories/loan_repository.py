from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from .base_repository import BaseRepository
from ..models.loan_application import LoanApplication

class LoanRepository(BaseRepository[LoanApplication]):
    def __init__(self, db: Session):
        super().__init__(db, LoanApplication)
    
    def get_by_email(self, email: str) -> Optional[LoanApplication]:
        return self.db.query(LoanApplication).filter(LoanApplication.email == email).first()
    
    def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[LoanApplication]:
        return (self.db.query(LoanApplication)
                .filter(LoanApplication.status == status)
                .offset(skip)
                .limit(limit)
                .all())
    
    def search_applications(self, 
                          name: Optional[str] = None,
                          email: Optional[str] = None,
                          status: Optional[str] = None,
                          approved: Optional[bool] = None,
                          skip: int = 0,
                          limit: int = 100) -> List[LoanApplication]:
        query = self.db.query(LoanApplication)
        
        if name:
            query = query.filter(LoanApplication.name.ilike(f"%{name}%"))
        if email:
            query = query.filter(LoanApplication.email.ilike(f"%{email}%"))
        if status:
            query = query.filter(LoanApplication.status == status)
        if approved is not None:
            query = query.filter(LoanApplication.is_approved == approved)
        
        return query.offset(skip).limit(limit).all()