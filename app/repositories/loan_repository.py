# app/repositories/loan_repository.py - Updated with pagination support
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from datetime import datetime
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
    
    def get_paginated_applications(self, 
                                 skip: int = 0,
                                 limit: int = 20,
                                 **filters) -> Tuple[List[LoanApplication], int]:
        """Get paginated loan applications with filters and total count"""
        
        # Build base query
        query = self.db.query(LoanApplication)
        
        # Apply filters
        query = self._apply_filters(query, **filters)
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply sorting
        query = self._apply_sorting(query, **filters)
        
        # Apply pagination
        applications = query.offset(skip).limit(limit).all()
        
        return applications, total_count
    
    def _apply_filters(self, query, **filters):
        """Apply filters to the query"""
        
        # Text filters
        if filters.get('name'):
            query = query.filter(LoanApplication.name.ilike(f"%{filters['name']}%"))
        
        if filters.get('email'):
            query = query.filter(LoanApplication.email.ilike(f"%{filters['email']}%"))
        
        # Status filters
        if filters.get('status'):
            query = query.filter(LoanApplication.status == filters['status'])
        
        if filters.get('approved') is not None:
            query = query.filter(LoanApplication.is_approved == filters['approved'])
        
        # Risk category filter
        if filters.get('risk_category'):
            risk_ranges = {
                'very_low': (0, 20),
                'low': (20, 40),
                'medium': (40, 60),
                'high': (60, 80),
                'very_high': (80, 100)
            }
            
            if filters['risk_category'] in risk_ranges:
                min_risk, max_risk = risk_ranges[filters['risk_category']]
                query = query.filter(
                    and_(
                        LoanApplication.risk_score >= min_risk,
                        LoanApplication.risk_score < max_risk
                    )
                )
        
        # Credit score range filters
        if filters.get('credit_score_min') is not None:
            query = query.filter(LoanApplication.credit_score >= filters['credit_score_min'])
        
        if filters.get('credit_score_max') is not None:
            query = query.filter(LoanApplication.credit_score <= filters['credit_score_max'])
        
        # Loan amount range filters
        if filters.get('loan_amount_min') is not None:
            query = query.filter(LoanApplication.loan_amount >= filters['loan_amount_min'])
        
        if filters.get('loan_amount_max') is not None:
            query = query.filter(LoanApplication.loan_amount <= filters['loan_amount_max'])
        
        # Date range filters
        if filters.get('created_after'):
            query = query.filter(LoanApplication.created_at >= filters['created_after'])
        
        if filters.get('created_before'):
            query = query.filter(LoanApplication.created_at <= filters['created_before'])
        
        # Model filter
        if filters.get('model_used'):
            query = query.filter(LoanApplication.model_used == filters['model_used'])
        
        # Industry filter
        if filters.get('industry'):
            query = query.filter(LoanApplication.industry == filters['industry'])
        
        # Property area filter
        if filters.get('property_area'):
            query = query.filter(LoanApplication.property_area == filters['property_area'])
        
        # City tier filter
        if filters.get('city_tier'):
            query = query.filter(LoanApplication.city_tier == filters['city_tier'])
        
        return query
    
    def _apply_sorting(self, query, **filters):
        """Apply sorting to the query"""
        
        sort_by = filters.get('sort_by', 'created_at')
        sort_order = filters.get('sort_order', 'desc')
        
        # Map sort fields to actual columns
        sort_mapping = {
            'created_at': LoanApplication.created_at,
            'updated_at': LoanApplication.updated_at,
            'loan_amount': LoanApplication.loan_amount,
            'credit_score': LoanApplication.credit_score,
            'risk_score': LoanApplication.risk_score,
            'confidence_score': LoanApplication.confidence_score,
            'applicant_income': LoanApplication.applicant_income,
            'name': LoanApplication.name,
            'email': LoanApplication.email,
            'age': LoanApplication.age
        }
        
        if sort_by in sort_mapping:
            column = sort_mapping[sort_by]
            if sort_order.lower() == 'asc':
                query = query.order_by(asc(column))
            else:
                query = query.order_by(desc(column))
        else:
            # Default sorting
            query = query.order_by(desc(LoanApplication.created_at))
        
        return query
    
    def search_applications(self, 
                          skip: int = 0, 
                          limit: int = 100,
                          **filters) -> List[LoanApplication]:
        """Search applications (legacy method for backward compatibility)"""
        applications, _ = self.get_paginated_applications(skip=skip, limit=limit, **filters)
        return applications
    
    def get_applications_by_date_range(self, 
                                     start_date: datetime, 
                                     end_date: datetime) -> Tuple[List[LoanApplication], int]:
        """Get applications within a date range"""
        
        query = self.db.query(LoanApplication).filter(
            and_(
                LoanApplication.created_at >= start_date,
                LoanApplication.created_at <= end_date
            )
        )
        
        total_count = query.count()
        applications = query.order_by(desc(LoanApplication.created_at)).all()
        
        return applications, total_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get general statistics about loan applications"""
        
        total_applications = self.db.query(func.count(LoanApplication.id)).scalar()
        
        approved_count = self.db.query(func.count(LoanApplication.id)).filter(
            LoanApplication.is_approved == True
        ).scalar()
        
        rejected_count = self.db.query(func.count(LoanApplication.id)).filter(
            LoanApplication.is_approved == False
        ).scalar()
        
        pending_count = self.db.query(func.count(LoanApplication.id)).filter(
            LoanApplication.is_approved.is_(None)
        ).scalar()
        
        # Average loan amount
        avg_loan_amount = self.db.query(func.avg(LoanApplication.loan_amount)).scalar() or 0
        
        # Average credit score
        avg_credit_score = self.db.query(func.avg(LoanApplication.credit_score)).filter(
            LoanApplication.credit_score.isnot(None)
        ).scalar() or 0
        
        # Average risk score
        avg_risk_score = self.db.query(func.avg(LoanApplication.risk_score)).filter(
            LoanApplication.risk_score.isnot(None)
        ).scalar() or 0
        
        return {
            'total_applications': total_applications,
            'approved_count': approved_count,
            'rejected_count': rejected_count,
            'pending_count': pending_count,
            'approval_rate': (approved_count / total_applications * 100) if total_applications > 0 else 0,
            'avg_loan_amount': float(avg_loan_amount),
            'avg_credit_score': float(avg_credit_score),
            'avg_risk_score': float(avg_risk_score)
        }
    
    def get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of applications by risk category"""
        
        risk_counts = {
            'very_low': 0,
            'low': 0,
            'medium': 0,
            'high': 0,
            'very_high': 0
        }
        
        # Query applications with risk scores
        applications = self.db.query(LoanApplication.risk_score).filter(
            LoanApplication.risk_score.isnot(None)
        ).all()
        
        for app in applications:
            risk_score = app.risk_score
            if risk_score <= 20:
                risk_counts['very_low'] += 1
            elif risk_score <= 40:
                risk_counts['low'] += 1
            elif risk_score <= 60:
                risk_counts['medium'] += 1
            elif risk_score <= 80:
                risk_counts['high'] += 1
            else:
                risk_counts['very_high'] += 1
        
        return risk_counts
    
    def get_model_usage_distribution(self) -> Dict[str, int]:
        """Get distribution of applications by model used"""
        
        result = self.db.query(
            LoanApplication.model_used,
            func.count(LoanApplication.id).label('count')
        ).filter(
            LoanApplication.model_used.isnot(None)
        ).group_by(LoanApplication.model_used).all()
        
        return {row.model_used: row.count for row in result}
    
    def get_approval_rate_by_model(self) -> Dict[str, float]:
        """Get approval rates by model type"""
        
        result = self.db.query(
            LoanApplication.model_used,
            func.count(LoanApplication.id).label('total'),
            func.sum(func.cast(LoanApplication.is_approved, LoanApplication.id.type)).label('approved')
        ).filter(
            and_(
                LoanApplication.model_used.isnot(None),
                LoanApplication.is_approved.isnot(None)
            )
        ).group_by(LoanApplication.model_used).all()
        
        approval_rates = {}
        for row in result:
            if row.total > 0:
                approval_rates[row.model_used] = (row.approved or 0) / row.total * 100
            else:
                approval_rates[row.model_used] = 0
        
        return approval_rates
    
    def get_recent_applications(self, limit: int = 10) -> List[LoanApplication]:
        """Get most recent applications"""
        
        return (self.db.query(LoanApplication)
                .order_by(desc(LoanApplication.created_at))
                .limit(limit)
                .all())
    
    def get_high_risk_applications(self, 
                                 risk_threshold: float = 70, 
                                 limit: int = 50) -> List[LoanApplication]:
        """Get applications with high risk scores"""
        
        return (self.db.query(LoanApplication)
                .filter(LoanApplication.risk_score >= risk_threshold)
                .order_by(desc(LoanApplication.risk_score))
                .limit(limit)
                .all())
    
    def bulk_update_status(self, application_ids: List[int], new_status: str) -> int:
        """Bulk update status for multiple applications"""
        
        updated_count = (self.db.query(LoanApplication)
                        .filter(LoanApplication.id.in_(application_ids))
                        .update(
                            {'status': new_status, 'updated_at': func.now()},
                            synchronize_session=False
                        ))
        
        self.db.commit()
        return updated_count
    
    def search_by_text(self, search_term: str, limit: int = 50) -> List[LoanApplication]:
        """Search applications by text (name, email)"""
        
        search_pattern = f"%{search_term}%"
        
        return (self.db.query(LoanApplication)
                .filter(
                    or_(
                        LoanApplication.name.ilike(search_pattern),
                        LoanApplication.email.ilike(search_pattern)
                    )
                )
                .order_by(desc(LoanApplication.created_at))
                .limit(limit)
                .all())