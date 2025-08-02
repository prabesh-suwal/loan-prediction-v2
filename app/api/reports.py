from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from ..config.database import get_db
from ..core.dependencies import get_current_user
from ..services.loan_service import LoanService
from ..utils.helpers import get_risk_category, format_currency

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("/dashboard")
async def get_dashboard_stats(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get dashboard statistics"""
    
    loan_service = LoanService(db)
    
    # Date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Get all applications in date range
        all_applications = loan_service.get_loan_applications(limit=10000)
        
        # Filter by date range
        recent_applications = [
            app for app in all_applications 
            if app.created_at >= start_date
        ]
        
        # Calculate statistics
        total_applications = len(recent_applications)
        approved_applications = len([app for app in recent_applications if app.is_approved])
        rejected_applications = len([app for app in recent_applications if app.is_approved == False])
        pending_applications = len([app for app in recent_applications if app.is_approved is None])
        
        # Approval rate
        approval_rate = (approved_applications / total_applications * 100) if total_applications > 0 else 0
        
        # Risk distribution
        risk_distribution = {"Very Low": 0, "Low": 0, "Medium": 0, "High": 0, "Very High": 0}
        for app in recent_applications:
            if app.risk_score is not None:
                risk_category = get_risk_category(app.risk_score)
                risk_distribution[risk_category] += 1
        
        # Total loan amounts
        total_requested = sum([app.loan_amount for app in recent_applications])
        total_approved = sum([app.loan_amount for app in recent_applications if app.is_approved])
        
        # Average processing metrics
        avg_risk_score = sum([app.risk_score for app in recent_applications if app.risk_score is not None]) / len([app for app in recent_applications if app.risk_score is not None]) if recent_applications else 0
        avg_confidence = sum([app.confidence_score for app in recent_applications if app.confidence_score is not None]) / len([app for app in recent_applications if app.confidence_score is not None]) if recent_applications else 0
        
        return {
            "period": f"Last {days} days",
            "summary": {
                "total_applications": total_applications,
                "approved_applications": approved_applications,
                "rejected_applications": rejected_applications,
                "pending_applications": pending_applications,
                "approval_rate": round(approval_rate, 2)
            },
            "financial": {
                "total_requested_amount": total_requested,
                "total_approved_amount": total_approved,
                "total_requested_formatted": format_currency(total_requested),
                "total_approved_formatted": format_currency(total_approved)
            },
            "risk_analysis": {
                "risk_distribution": risk_distribution,
                "average_risk_score": round(avg_risk_score, 2),
                "average_confidence_score": round(avg_confidence, 2)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard stats: {str(e)}")

@router.get("/trends")
async def get_approval_trends(
    days: int = Query(30, ge=7, le=365),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get approval trends over time"""
    
    loan_service = LoanService(db)
    
    try:
        # Get applications
        all_applications = loan_service.get_loan_applications(limit=10000)
        
        # Group by date
        daily_stats = {}
        
        for app in all_applications:
            date_key = app.created_at.date().isoformat()
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "date": date_key,
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                    "pending": 0
                }
            
            daily_stats[date_key]["total"] += 1
            
            if app.is_approved is True:
                daily_stats[date_key]["approved"] += 1
            elif app.is_approved is False:
                daily_stats[date_key]["rejected"] += 1
            else:
                daily_stats[date_key]["pending"] += 1
        
        # Convert to list and sort by date
        trends = list(daily_stats.values())
        trends.sort(key=lambda x: x["date"])
        
        # Calculate approval rates
        for stat in trends:
            if stat["total"] > 0:
                stat["approval_rate"] = round((stat["approved"] / stat["total"]) * 100, 2)
            else:
                stat["approval_rate"] = 0
        
        return {
            "period": f"Last {days} days",
            "trends": trends[-days:]  # Get last N days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating trends: {str(e)}")

@router.get("/risk-analysis")
async def get_risk_analysis(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detailed risk analysis"""
    
    loan_service = LoanService(db)
    
    try:
        # Get all processed applications
        all_applications = loan_service.get_loan_applications(limit=10000)
        processed_applications = [app for app in all_applications if app.risk_score is not None]
        
        if not processed_applications:
            return {"message": "No processed applications found for risk analysis"}
        
        # Risk score distribution
        risk_ranges = {
            "0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0
        }
        
        for app in processed_applications:
            score = app.risk_score
            if score <= 20:
                risk_ranges["0-20"] += 1
            elif score <= 40:
                risk_ranges["21-40"] += 1
            elif score <= 60:
                risk_ranges["41-60"] += 1
            elif score <= 80:
                risk_ranges["61-80"] += 1
            else:
                risk_ranges["81-100"] += 1
        
        # Risk vs Approval correlation
        risk_approval_correlation = {}
        for app in processed_applications:
            risk_category = get_risk_category(app.risk_score)
            if risk_category not in risk_approval_correlation:
                risk_approval_correlation[risk_category] = {"total": 0, "approved": 0}
            
            risk_approval_correlation[risk_category]["total"] += 1
            if app.is_approved:
                risk_approval_correlation[risk_category]["approved"] += 1
        
        # Calculate approval rates by risk category
        for category in risk_approval_correlation:
            total = risk_approval_correlation[category]["total"]
            approved = risk_approval_correlation[category]["approved"]
            risk_approval_correlation[category]["approval_rate"] = round((approved / total) * 100, 2) if total > 0 else 0
        
        # Average risk scores by approval status
        approved_apps = [app for app in processed_applications if app.is_approved]
        rejected_apps = [app for app in processed_applications if app.is_approved == False]
        
        avg_risk_approved = sum([app.risk_score for app in approved_apps]) / len(approved_apps) if approved_apps else 0
        avg_risk_rejected = sum([app.risk_score for app in rejected_apps]) / len(rejected_apps) if rejected_apps else 0
        
        return {
            "total_processed_applications": len(processed_applications),
            "risk_score_distribution": risk_ranges,
            "risk_vs_approval": risk_approval_correlation,
            "average_risk_scores": {
                "approved_applications": round(avg_risk_approved, 2),
                "rejected_applications": round(avg_risk_rejected, 2)
            },
            "insights": [
                f"Average risk score for approved loans: {avg_risk_approved:.1f}",
                f"Average risk score for rejected loans: {avg_risk_rejected:.1f}",
                f"Risk difference: {abs(avg_risk_rejected - avg_risk_approved):.1f} points"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating risk analysis: {str(e)}")