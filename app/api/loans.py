from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional
from ..config.database import get_db
from ..schemas.loan_application import (
    LoanApplicationCreate, 
    LoanApplicationResponse, 
    LoanApplicationDetailedResponse,
    PaginatedLoanResponse,
    PredictionResult,
    DecisionExplanation
)
from ..services.loan_service import LoanService
from ..core.dependencies import get_current_user

router = APIRouter(prefix="/loans", tags=["loans"])



@router.post("/", response_model=LoanApplicationResponse)
async def create_loan_application(
    loan_data: LoanApplicationCreate,
    db: Session = Depends(get_db)
):
    """Create new loan application and get ML prediction with explanation"""
    
    loan_service = LoanService(db)
    
    try:
        # Extract model preference
        preferred_model = loan_data.preferred_model
        # Create loan application data without the model preference
        loan_create_data = LoanApplicationCreate(**loan_data.dict(exclude={'preferred_model'}))

        # Create application with specified model
        loan_application = loan_service.create_loan_application(loan_create_data, preferred_model)
        
        # Build prediction result with explanation
        prediction_result = None
        if loan_application.is_approved is not None:
            prediction_result = PredictionResult(
                approved=loan_application.is_approved,
                confidence_score=loan_application.confidence_score,
                risk_score=loan_application.risk_score,
                prediction_details=loan_application.prediction_details,
                recommended_interest_rate=loan_application.recommended_interest_rate,
                conditions=loan_application.conditions or [],
                explanation=DecisionExplanation(
                    decision_summary=loan_application.decision_summary,
                    detailed_explanation=loan_application.detailed_explanation,
                    risk_assessment=loan_application.risk_explanation,
                    key_factors=loan_application.key_factors,
                    recommendations=loan_application.recommendations,
                    next_steps=loan_application.next_steps,
                    explanation_metadata=loan_application.explanation_metadata
                ) if loan_application.decision_summary else None
            )
        
        return LoanApplicationResponse(
            id=loan_application.id,
            status=loan_application.status,
            prediction_result=prediction_result,
            created_at=loan_application.created_at,
            updated_at=loan_application.updated_at,
            model_used=loan_application.model_used,
            decision_summary=loan_application.decision_summary,
            plain_text_summary=loan_application.plain_text_summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing loan application: {str(e)}")


@router.get("/", response_model=PaginatedLoanResponse)
async def get_loan_applications(
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    name: Optional[str] = Query(None, description="Filter by applicant name"),
    email: Optional[str] = Query(None, description="Filter by email"),
    status: Optional[str] = Query(None, description="Filter by status (pending, processed, approved, rejected)"),
    approved: Optional[bool] = Query(None, description="Filter by approval status"),
    risk_category: Optional[str] = Query(None, description="Filter by risk category (very_low, low, medium, high, very_high)"),
    credit_score_min: Optional[int] = Query(None, ge=300, le=850, description="Minimum credit score"),
    credit_score_max: Optional[int] = Query(None, ge=300, le=850, description="Maximum credit score"),
    loan_amount_min: Optional[float] = Query(None, ge=0, description="Minimum loan amount"),
    loan_amount_max: Optional[float] = Query(None, ge=0, description="Maximum loan amount"),
    sort_by: Optional[str] = Query("created_at", description="Sort by field (created_at, loan_amount, credit_score, risk_score)"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get paginated list of loan applications with filters and search"""
    
    loan_service = LoanService(db)
    
    # Convert page-based to skip-based pagination
    skip = (page - 1) * page_size
    limit = page_size
    
    # Build filters
    filters = {}
    if name:
        filters['name'] = name
    if email:
        filters['email'] = email
    if status:
        filters['status'] = status
    if approved is not None:
        filters['approved'] = approved
    if risk_category:
        filters['risk_category'] = risk_category
    if credit_score_min is not None:
        filters['credit_score_min'] = credit_score_min
    if credit_score_max is not None:
        filters['credit_score_max'] = credit_score_max
    if loan_amount_min is not None:
        filters['loan_amount_min'] = loan_amount_min
    if loan_amount_max is not None:
        filters['loan_amount_max'] = loan_amount_max
    if sort_by:
        filters['sort_by'] = sort_by
    if sort_order:
        filters['sort_order'] = sort_order
    
    # Get applications with total count
    loan_applications, total_count = loan_service.get_loan_applications_paginated(
        skip=skip, 
        limit=limit, 
        **filters
    )
    
    # Convert to response format
    loan_responses = []
    for app in loan_applications:
        prediction_result = None
        if app.is_approved is not None:
            prediction_result = PredictionResult(
                approved=app.is_approved,
                confidence_score=app.confidence_score,
                risk_score=app.risk_score,
                prediction_details=app.prediction_details,
                recommended_interest_rate=app.recommended_interest_rate,
                conditions=app.conditions or []
                # Note: Not including full explanation in list view for performance
            )
        
        loan_responses.append(LoanApplicationResponse(
            id=app.id,
            status=app.status,
            prediction_result=prediction_result,
            created_at=app.created_at,
            updated_at=app.updated_at,
            model_used=app.model_used,
            decision_summary=app.decision_summary,
            plain_text_summary=app.plain_text_summary
        ))
    
    # Calculate pagination metadata
    total_pages = (total_count + page_size - 1) // page_size
    has_next = page < total_pages
    has_previous = page > 1
    
    return PaginatedLoanResponse(
        data=loan_responses,
        pagination={
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_previous": has_previous,
            "next_page": page + 1 if has_next else None,
            "previous_page": page - 1 if has_previous else None
        },
        filters=filters,
        summary={
            "total_applications": total_count,
            "approved_count": len([app for app in loan_applications if app.is_approved]),
            "rejected_count": len([app for app in loan_applications if app.is_approved == False]),
            "pending_count": len([app for app in loan_applications if app.is_approved is None])
        }
    )


@router.get("/{loan_id}", response_model=LoanApplicationDetailedResponse)
async def get_loan_application(
    loan_id: int = Path(..., description="Loan application ID"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get specific loan application with detailed explanation"""
    
    loan_service = LoanService(db)
    loan_application = loan_service.get_loan_application(loan_id)
    
    if not loan_application:
        raise HTTPException(status_code=404, detail="Loan application not found")
    
    # Build detailed prediction result with full explanation
    prediction_result = None
    if loan_application.is_approved is not None:
        prediction_result = PredictionResult(
            approved=loan_application.is_approved,
            confidence_score=loan_application.confidence_score,
            risk_score=loan_application.risk_score,
            prediction_details=loan_application.prediction_details,
            recommended_interest_rate=loan_application.recommended_interest_rate,
            conditions=loan_application.conditions or [],
            explanation=DecisionExplanation(
                decision_summary=loan_application.decision_summary,
                detailed_explanation=loan_application.detailed_explanation,
                risk_assessment=loan_application.risk_explanation,
                key_factors=loan_application.key_factors,
                recommendations=loan_application.recommendations,
                next_steps=loan_application.next_steps,
                explanation_metadata=loan_application.explanation_metadata
            ) if loan_application.decision_summary else None
        )
    
    return LoanApplicationDetailedResponse(
        id=loan_application.id,
        status=loan_application.status,
        prediction_result=prediction_result,
        created_at=loan_application.created_at,
        updated_at=loan_application.updated_at,
        model_used=loan_application.model_used,
        decision_summary=loan_application.decision_summary,
        plain_text_summary=loan_application.plain_text_summary,
        detailed_explanation=loan_application.detailed_explanation,
        risk_explanation=loan_application.risk_explanation,
        key_factors=loan_application.key_factors,
        recommendations=loan_application.recommendations,
        next_steps=loan_application.next_steps,
        explanation_metadata=loan_application.explanation_metadata
    )

@router.get("/{loan_id}/explanation")
async def get_loan_explanation(
    loan_id: int = Path(..., description="Loan application ID"),
    format: str = Query("json", regex="^(json|text|summary)$", description="Response format"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detailed explanation for a loan decision in different formats"""
    
    loan_service = LoanService(db)
    loan_application = loan_service.get_loan_application(loan_id)
    
    if not loan_application:
        raise HTTPException(status_code=404, detail="Loan application not found")
    
    if loan_application.is_approved is None:
        raise HTTPException(status_code=400, detail="Loan application not yet processed")
    
    if format == "text":
        # Return plain text summary
        return {
            "format": "text",
            "explanation": loan_application.plain_text_summary or "No explanation available"
        }
    
    elif format == "summary":
        # Return quick summary
        summary = loan_service.get_explanation_summary(loan_id)
        return {
            "format": "summary",
            "explanation": summary
        }
    
    else:  # json format (default)
        # Return full structured explanation
        explanation = {
            "decision_summary": loan_application.decision_summary,
            "detailed_explanation": loan_application.detailed_explanation,
            "risk_assessment": loan_application.risk_explanation,
            "key_factors": loan_application.key_factors,
            "recommendations": loan_application.recommendations,
            "next_steps": loan_application.next_steps,
            "explanation_metadata": loan_application.explanation_metadata
        }
        
        return {
            "format": "json",
            "explanation": explanation
        }

@router.post("/{loan_id}/regenerate-explanation")
async def regenerate_explanation(
    loan_id: int = Path(..., description="Loan application ID"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Regenerate explanation for an existing loan application"""
    
    loan_service = LoanService(db)
    
    try:
        updated_application = loan_service.regenerate_explanation(loan_id)
        
        if not updated_application:
            raise HTTPException(status_code=404, detail="Loan application not found")
        
        return {
            "message": "Explanation regenerated successfully",
            "loan_id": loan_id,
            "decision_summary": updated_application.decision_summary,
            "updated_at": updated_application.updated_at
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerating explanation: {str(e)}")

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

@router.get("/explanations/analytics")
async def get_explanation_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get analytics on explanations and decision patterns"""
    
    loan_service = LoanService(db)
    
    try:
        # Get recent applications
        applications = loan_service.get_loan_applications(limit=1000)
        
        # Filter processed applications with explanations
        processed_apps = [app for app in applications if app.is_approved is not None and app.explanation_metadata]
        
        if not processed_apps:
            return {"message": "No processed applications with explanations found"}
        
        # Analyze explanation patterns
        analytics = {
            "total_processed": len(processed_apps),
            "approval_rate": sum(1 for app in processed_apps if app.is_approved) / len(processed_apps) * 100,
            "confidence_distribution": {},
            "risk_distribution": {},
            "common_risk_factors": {},
            "common_positive_factors": {},
            "average_recommendations_per_application": 0
        }
        
        # Analyze confidence levels
        confidence_levels = {}
        risk_categories = {}
        all_risk_factors = []
        all_positive_factors = []
        total_recommendations = 0
        
        for app in processed_apps:
            # Confidence analysis
            if app.explanation_metadata and 'confidence_level' in app.explanation_metadata:
                conf_level = app.explanation_metadata['confidence_level']
                confidence_levels[conf_level] = confidence_levels.get(conf_level, 0) + 1
            
            # Risk analysis
            if app.explanation_metadata and 'risk_category' in app.explanation_metadata:
                risk_cat = app.explanation_metadata['risk_category']
                risk_categories[risk_cat] = risk_categories.get(risk_cat, 0) + 1
            
            # Factor analysis
            if app.key_factors:
                if 'risk_factors' in app.key_factors:
                    all_risk_factors.extend(app.key_factors['risk_factors'])
                if 'positive_factors' in app.key_factors:
                    all_positive_factors.extend(app.key_factors['positive_factors'])
            
            # Recommendations count
            if app.recommendations:
                total_recommendations += len(app.recommendations)
        
        # Count factor frequencies
        risk_factor_counts = {}
        for factor in all_risk_factors:
            risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
        
        positive_factor_counts = {}
        for factor in all_positive_factors:
            positive_factor_counts[factor] = positive_factor_counts.get(factor, 0) + 1
        
        # Update analytics
        analytics.update({
            "confidence_distribution": confidence_levels,
            "risk_distribution": risk_categories,
            "common_risk_factors": dict(sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "common_positive_factors": dict(sorted(positive_factor_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "average_recommendations_per_application": total_recommendations / len(processed_apps) if processed_apps else 0
        })
        
        return analytics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation analytics: {str(e)}")