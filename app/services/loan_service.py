from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from ..repositories.loan_repository import LoanRepository
from ..repositories.weight_repository import WeightRepository
from ..schemas.loan_application import LoanApplicationCreate, PredictionResult
from ..models.loan_application import LoanApplication
from .ml_service import MLService
from ..ml.decision_explainer import LoanDecisionExplainer

class LoanService:
    def __init__(self, db: Session):
        self.db = db
        self.loan_repository = LoanRepository(db)
        weight_repository = WeightRepository(db)
        self.ml_service = MLService(weight_repository)
        self.explainer = LoanDecisionExplainer()
    
    def create_loan_application(self, loan_data: LoanApplicationCreate, model_type: str = None) -> LoanApplication:
        """Create new loan application with ML prediction using specified model"""
        
        # Switch model if specified
        if model_type and model_type != self.ml_service.current_model_type:
            try:
                self.ml_service.switch_model(model_type)
            except Exception as e:
                # Log warning but continue with current model
                print(f"Warning: Could not switch to {model_type}, using {self.ml_service.current_model_type}: {e}")
        
        # Convert Pydantic model to dict
        loan_dict = loan_data.dict()
        
        # Get ML prediction
        prediction_result = self.ml_service.predict_loan_approval(loan_dict)

        # Generate human-readable explanation
        explanation = self.explainer.generate_explanation(loan_dict, prediction_result)
        
        # Generate plain text summary for easy display
        plain_text_summary = self.explainer.generate_plain_text_summary(explanation)
        
        # Add prediction results to loan data
        loan_dict.update({
            'is_approved': prediction_result['approved'],
            'confidence_score': prediction_result['confidence_score'],
            'risk_score': prediction_result['risk_score'],
            'prediction_details': prediction_result['prediction_details'],
            'recommended_interest_rate': prediction_result['recommended_interest_rate'],
            'conditions': prediction_result['conditions'],
            'status': 'processed',
            'model_used': prediction_result.get('model_used'),
            'model_performance': prediction_result.get('model_performance'),
            
            # Add explanation fields
            'decision_summary': explanation['decision_summary'],
            'detailed_explanation': explanation['detailed_explanation'],
            'risk_explanation': explanation['risk_assessment'],
            'key_factors': explanation['key_factors'],
            'recommendations': explanation['recommendations'],
            'next_steps': explanation['next_steps'],
            'explanation_metadata': explanation['explanation_metadata'],
            'plain_text_summary': plain_text_summary
        })
        
        # Remove the key
        loan_dict.pop('preferred_model', None)
        # Save to database
        loan_application = self.loan_repository.create(loan_dict)
        
        return loan_application
    
    def get_loan_application(self, loan_id: int) -> Optional[LoanApplication]:
        """Get loan application by ID"""
        return self.loan_repository.get_by_id(loan_id)
    
    def get_loan_applications(self, 
                            skip: int = 0, 
                            limit: int = 100,
                            **filters) -> List[LoanApplication]:
        """Get paginated list of loan applications with filters"""
        return self.loan_repository.search_applications(skip=skip, limit=limit, **filters)
    
    def get_loan_applications_paginated(self, 
                                      skip: int = 0, 
                                      limit: int = 20,
                                      **filters) -> Tuple[List[LoanApplication], int]:
        """Get paginated list of loan applications with filters and total count"""
        return self.loan_repository.get_paginated_applications(skip=skip, limit=limit, **filters)
    
    def update_loan_status(self, loan_id: int, status: str) -> Optional[LoanApplication]:
        """Update loan application status"""
        return self.loan_repository.update(loan_id, {'status': status})
    
    def regenerate_explanation(self, loan_id: int) -> Optional[LoanApplication]:
        """Regenerate explanation for an existing loan application"""
        
        loan_application = self.loan_repository.get_by_id(loan_id)
        if not loan_application:
            return None
        
        # Reconstruct loan data from database record
        loan_dict = {
            'name': loan_application.name,
            'email': loan_application.email,
            'gender': loan_application.gender,
            'married': loan_application.married,
            'dependents': loan_application.dependents,
            'education': loan_application.education,
            'age': loan_application.age,
            'children': loan_application.children,
            'spouse_employed': loan_application.spouse_employed,
            'self_employed': loan_application.self_employed,
            'employment_type': loan_application.employment_type,
            'years_in_current_job': loan_application.years_in_current_job,
            'employer_category': loan_application.employer_category,
            'industry': loan_application.industry,
            'applicant_income': loan_application.applicant_income,
            'coapplicant_income': loan_application.coapplicant_income,
            'monthly_expenses': loan_application.monthly_expenses,
            'other_emis': loan_application.other_emis,
            'loan_amount': loan_application.loan_amount,
            'loan_amount_term': loan_application.loan_amount_term,
            'loan_purpose': loan_application.loan_purpose,
            'requested_interest_rate': loan_application.requested_interest_rate,
            'credit_score': loan_application.credit_score,
            'credit_history': loan_application.credit_history,
            'no_of_credit_cards': loan_application.no_of_credit_cards,
            'loan_default_history': loan_application.loan_default_history,
            'avg_payment_delay_days': loan_application.avg_payment_delay_days,
            'has_vehicle': loan_application.has_vehicle,
            'has_life_insurance': loan_application.has_life_insurance,
            'property_area': loan_application.property_area,
            'bank_account_type': loan_application.bank_account_type,
            'bank_balance': loan_application.bank_balance,
            'savings_score': loan_application.savings_score,
            'collateral_type': loan_application.collateral_type,
            'collateral_value': loan_application.collateral_value,
            'city_tier': loan_application.city_tier,
            'pincode': loan_application.pincode,
            'region_default_rate': loan_application.region_default_rate
        }
        
        # Reconstruct prediction result
        prediction_result = {
            'approved': loan_application.is_approved,
            'confidence_score': loan_application.confidence_score,
            'risk_score': loan_application.risk_score,
            'prediction_details': loan_application.prediction_details or {},
            'recommended_interest_rate': loan_application.recommended_interest_rate,
            'conditions': loan_application.conditions or []
        }
        
        # Generate new explanation
        explanation = self.explainer.generate_explanation(loan_dict, prediction_result)
        plain_text_summary = self.explainer.generate_plain_text_summary(explanation)
        
        # Update with new explanation
        update_data = {
            'decision_summary': explanation['decision_summary'],
            'detailed_explanation': explanation['detailed_explanation'],
            'risk_explanation': explanation['risk_assessment'],
            'key_factors': explanation['key_factors'],
            'recommendations': explanation['recommendations'],
            'next_steps': explanation['next_steps'],
            'explanation_metadata': explanation['explanation_metadata'],
            'plain_text_summary': plain_text_summary
        }
        
        return self.loan_repository.update(loan_id, update_data)
    
    def get_model_usage_stats(self) -> Dict[str, Any]:
        """Get statistics on model usage"""
        return {
            'current_model': self.ml_service.current_model_type,
            'available_models': self.ml_service.get_available_models(),
            'model_comparison': self.ml_service.get_model_comparison()
        }
    
    def get_explanation_summary(self, loan_id: int) -> Optional[Dict[str, Any]]:
        """Get a summary of the explanation for quick reference"""
        
        loan_application = self.loan_repository.get_by_id(loan_id)
        if not loan_application:
            return None
        
        return {
            'loan_id': loan_id,
            'decision': 'Approved' if loan_application.is_approved else 'Rejected',
            'confidence_score': loan_application.confidence_score,
            'risk_score': loan_application.risk_score,
            'decision_summary': loan_application.decision_summary,
            'key_recommendations': (loan_application.recommendations or [])[:3],  # Top 3
            'next_step': (loan_application.next_steps or [None])[0],  # First step
            'risk_category': loan_application.explanation_metadata.get('risk_category') if loan_application.explanation_metadata else None,
            'confidence_level': loan_application.explanation_metadata.get('confidence_level') if loan_application.explanation_metadata else None
        }
    
    def get_loan_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get loan application statistics"""
        
        from datetime import datetime, timedelta
        
        # Date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get applications in date range with total count
        applications, total_count = self.loan_repository.get_applications_by_date_range(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate statistics
        total_applications = len(applications)
        approved_applications = len([app for app in applications if app.is_approved])
        rejected_applications = len([app for app in applications if app.is_approved == False])
        pending_applications = len([app for app in applications if app.is_approved is None])
        
        # Approval rate
        approval_rate = (approved_applications / total_applications * 100) if total_applications > 0 else 0
        
        # Average amounts
        if applications:
            avg_loan_amount = sum([app.loan_amount for app in applications]) / len(applications)
            avg_income = sum([app.applicant_income + (app.coapplicant_income or 0) for app in applications]) / len(applications)
            avg_credit_score = sum([app.credit_score for app in applications if app.credit_score]) / len([app for app in applications if app.credit_score])
        else:
            avg_loan_amount = avg_income = avg_credit_score = 0
        
        # Risk distribution
        risk_distribution = {"very_low": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0}
        for app in applications:
            if app.risk_score is not None:
                if app.risk_score <= 20:
                    risk_distribution["very_low"] += 1
                elif app.risk_score <= 40:
                    risk_distribution["low"] += 1
                elif app.risk_score <= 60:
                    risk_distribution["medium"] += 1
                elif app.risk_score <= 80:
                    risk_distribution["high"] += 1
                else:
                    risk_distribution["very_high"] += 1
        
        return {
            "period": f"Last {days} days",
            "overview": {
                "total_applications": total_applications,
                "approved_applications": approved_applications,
                "rejected_applications": rejected_applications,
                "pending_applications": pending_applications,
                "approval_rate": round(approval_rate, 2)
            },
            "averages": {
                "loan_amount": round(avg_loan_amount, 2),
                "income": round(avg_income, 2),
                "credit_score": round(avg_credit_score, 2)
            },
            "risk_distribution": risk_distribution,
            "trends": self._calculate_trends(applications, days)
        }
    
    
    def _calculate_trends(self, applications: List[LoanApplication], days: int) -> Dict[str, Any]:
        """Calculate trends from applications data"""
        
        if not applications:
            return {}
        
        # Group by date
        from collections import defaultdict
        daily_stats = defaultdict(lambda: {"total": 0, "approved": 0, "rejected": 0})
        
        for app in applications:
            date_key = app.created_at.date().isoformat()
            daily_stats[date_key]["total"] += 1
            
            if app.is_approved is True:
                daily_stats[date_key]["approved"] += 1
            elif app.is_approved is False:
                daily_stats[date_key]["rejected"] += 1
        
        # Convert to list format
        trends = []
        for date_str, stats in sorted(daily_stats.items()):
            approval_rate = (stats["approved"] / stats["total"] * 100) if stats["total"] > 0 else 0
            trends.append({
                "date": date_str,
                "total": stats["total"],
                "approved": stats["approved"],
                "rejected": stats["rejected"],
                "approval_rate": round(approval_rate, 2)
            })
        
        return {
            "daily_trends": trends[-30:],  # Last 30 days
            "average_daily_applications": sum([t["total"] for t in trends]) / len(trends) if trends else 0
        }