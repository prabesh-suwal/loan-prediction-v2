from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from ..config.database import get_db
from ..schemas.loan_application import LoanApplicationCreate, LoanApplicationResponse
from ..services.loan_service import LoanService
from ..core.dependencies import get_current_user

router = APIRouter(prefix="/loans", tags=["loans"])



@router.post("/", response_model=LoanApplicationResponse)
async def create_loan_application(
    loan_data: LoanApplicationCreate,
    db: Session = Depends(get_db)
):
    """Create new loan application and get ML prediction"""
    
    loan_service = LoanService(db)
    
    try:
        # Extract model preference
        preferred_model = loan_data.preferred_model
        # Create loan application data without the model preference
        loan_create_data = LoanApplicationCreate(**loan_data.dict(exclude={'preferred_model'}))

        # Create application with specified model
        loan_application = loan_service.create_loan_application(loan_create_data, preferred_model)
        # loan_application = loan_service.create_loan_application(loan_data)
        
        return LoanApplicationResponse(
            id=loan_application.id,
            status=loan_application.status,
            prediction_result={
                'approved': loan_application.is_approved,
                'confidence_score': loan_application.confidence_score,
                'risk_score': loan_application.risk_score,
                'prediction_details': loan_application.prediction_details,
                'recommended_interest_rate': loan_application.recommended_interest_rate,
                'conditions': loan_application.conditions or []
            },
            created_at=loan_application.created_at,
            updated_at=loan_application.updated_at,
            model_used=getattr(loan_application, 'model_used', None)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing loan application: {str(e)}")

@router.get("/", response_model=List[LoanApplicationResponse])
async def get_loan_applications(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    name: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    approved: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get paginated list of loan applications with filters"""
    
    loan_service = LoanService(db)
    
    filters = {}
    if name:
        filters['name'] = name
    if email:
        filters['email'] = email
    if status:
        filters['status'] = status
    if approved is not None:
        filters['approved'] = approved
    
    loan_applications = loan_service.get_loan_applications(
        skip=skip, 
        limit=limit, 
        **filters
    )
    
    return [
        LoanApplicationResponse(
            id=app.id,
            status=app.status,
            prediction_result={
                'approved': app.is_approved,
                'confidence_score': app.confidence_score,
                'risk_score': app.risk_score,
                'prediction_details': app.prediction_details,
                'recommended_interest_rate': app.recommended_interest_rate,
                'conditions': app.conditions or []
            } if app.is_approved is not None else None,
            created_at=app.created_at,
            updated_at=app.updated_at
        )
        for app in loan_applications
    ]

@router.get("/{loan_id}", response_model=LoanApplicationResponse)
async def get_loan_application(
    loan_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get specific loan application details"""
    
    loan_service = LoanService(db)
    loan_application = loan_service.get_loan_application(loan_id)
    
    if not loan_application:
        raise HTTPException(status_code=404, detail="Loan application not found")
    
    return LoanApplicationResponse(
        id=loan_application.id,
        status=loan_application.status,
        prediction_result={
            'approved': loan_application.is_approved,
            'confidence_score': loan_application.confidence_score,
            'risk_score': loan_application.risk_score,
            'prediction_details': loan_application.prediction_details,
            'recommended_interest_rate': loan_application.recommended_interest_rate,
            'conditions': loan_application.conditions or []
        } if loan_application.is_approved is not None else None,
        created_at=loan_application.created_at,
        updated_at=loan_application.updated_at
    )


@router.get("/model-stats")
async def get_model_usage_stats(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get model usage statistics"""
    
    loan_service = LoanService(db)
    
    try:
        stats = loan_service.get_model_usage_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model stats: {str(e)}")