import re
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def generate_application_id(email: str, timestamp: datetime = None) -> str:
    """Generate unique application ID"""
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    # Create hash from email and timestamp
    hash_input = f"{email}_{timestamp.isoformat()}"
    hash_object = hashlib.md5(hash_input.encode())
    hash_hex = hash_object.hexdigest()
    
    # Format as application ID
    return f"LOAN_{hash_hex[:10].upper()}"

def calculate_emi(principal: float, rate_annual: float, tenure_months: float) -> float:
    """Calculate EMI (Equated Monthly Installment)"""
    if rate_annual == 0:
        return principal / tenure_months
    
    rate_monthly = rate_annual / (12 * 100)
    emi = (principal * rate_monthly * (1 + rate_monthly) ** tenure_months) / \
          ((1 + rate_monthly) ** tenure_months - 1)
    
    return round(emi, 2)

def calculate_loan_to_value_ratio(loan_amount: float, collateral_value: float) -> float:
    """Calculate Loan-to-Value ratio"""
    if collateral_value == 0:
        return 100.0
    return min(100.0, (loan_amount / collateral_value) * 100)

def format_currency(amount: float, currency: str = "INR") -> str:
    """Format amount as currency"""
    if currency == "INR":
        # Indian currency formatting
        return f"₹{amount:,.2f}"
    else:
        return f"${amount:,.2f}"

def mask_sensitive_data(data: Dict[str, Any], fields_to_mask: List[str] = None) -> Dict[str, Any]:
    """Mask sensitive fields in data"""
    if fields_to_mask is None:
        fields_to_mask = ['password', 'hashed_password', 'bank_balance', 'collateral_value']
    
    masked_data = data.copy()
    for field in fields_to_mask:
        if field in masked_data:
            if isinstance(masked_data[field], str):
                masked_data[field] = "***"
            else:
                masked_data[field] = 0
    
    return masked_data

def calculate_age_from_date(birth_date: datetime) -> int:
    """Calculate age from birth date"""
    today = datetime.now()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def get_risk_category(risk_score: float) -> str:
    """Get risk category based on score"""
    if risk_score >= 80:
        return "Very High"
    elif risk_score >= 60:
        return "High"
    elif risk_score >= 40:
        return "Medium"
    elif risk_score >= 20:
        return "Low"
    else:
        return "Very Low"

def paginate_response(items: List[Any], page: int, page_size: int, total_count: int) -> Dict[str, Any]:
    """Create paginated response"""
    total_pages = (total_count + page_size - 1) // page_size
    
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }