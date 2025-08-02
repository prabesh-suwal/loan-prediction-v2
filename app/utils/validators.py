import re
from typing import Optional
from datetime import datetime

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate Indian phone number format"""
    pattern = r'^[6-9]\d{9}$'
    return re.match(pattern, phone) is not None

def validate_pan(pan: str) -> bool:
    """Validate PAN card format"""
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    return re.match(pattern, pan) is not None

def validate_aadhar(aadhar: str) -> bool:
    """Validate Aadhar number format"""
    pattern = r'^\d{12}$'
    return re.match(pattern, aadhar) is not None

def validate_pincode(pincode: str) -> bool:
    """Validate Indian pincode format"""
    pattern = r'^[1-9][0-9]{5}$'
    return re.match(pattern, pincode) is not None

def validate_ifsc(ifsc: str) -> bool:
    """Validate IFSC code format"""
    pattern = r'^[A-Z]{4}0[A-Z0-9]{6}$'
    return re.match(pattern, ifsc) is not None

def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"

def validate_income_consistency(applicant_income: float, 
                              coapplicant_income: float,
                              monthly_expenses: float,
                              other_emis: float) -> tuple[bool, str]:
    """Validate income and expense consistency"""
    total_income = applicant_income + coapplicant_income
    total_obligations = monthly_expenses + other_emis
    
    if total_obligations > total_income:
        return False, "Total monthly obligations exceed total income"
    
    # Check if expenses are reasonable (not more than 80% of income)
    if total_obligations > (total_income * 0.8):
        return False, "Monthly obligations are too high relative to income"
    
    return True, "Income and expenses are consistent"

def validate_loan_eligibility(loan_amount: float,
                            total_income: float,
                            existing_emis: float,
                            loan_tenure_months: float) -> tuple[bool, str]:
    """Basic loan eligibility validation"""
    
    # Calculate proposed EMI (assuming 12% interest rate for validation)
    if loan_tenure_months > 0:
        monthly_rate = 0.12 / 12
        proposed_emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_tenure_months) / \
                      ((1 + monthly_rate) ** loan_tenure_months - 1)
    else:
        return False, "Invalid loan tenure"
    
    total_emis = existing_emis + proposed_emi
    
    # EMI should not exceed 50% of income
    if total_emis > (total_income * 0.5):
        return False, "Proposed loan EMI exceeds affordability limits"
    
    # Minimum income requirement
    if total_income < 20000:
        return False, "Minimum income requirement not met"
    
    return True, "Basic eligibility criteria met"

def validate_credit_score_range(credit_score: int) -> tuple[bool, str]:
    """Validate credit score range"""
    if credit_score < 300 or credit_score > 850:
        return False, "Credit score must be between 300 and 850"
    
    if credit_score < 600:
        return True, "Credit score is below average - loan approval may be difficult"
    elif credit_score < 700:
        return True, "Credit score is fair - moderate approval chances"
    elif credit_score < 750:
        return True, "Credit score is good - high approval chances"
    else:
        return True, "Excellent credit score - very high approval chances"