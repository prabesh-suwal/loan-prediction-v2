from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.sql import func
from ..config.database import Base

class LoanApplication(Base):
    __tablename__ = "loan_applications"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Personal Info
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=False)
    married = Column(String(10), nullable=False)
    dependents = Column(Integer, nullable=False)
    education = Column(String(20), nullable=False)
    age = Column(Integer, nullable=False)
    children = Column(Integer, default=0)
    spouse_employed = Column(Boolean, default=False)
    
    # Employment
    self_employed = Column(String(10), nullable=False)
    employment_type = Column(String(20), default="Salaried")
    years_in_current_job = Column(Float, default=2.0)
    employer_category = Column(String(10), default="B")
    industry = Column(String(20), default="Others")
    
    # Financial
    applicant_income = Column(Float, nullable=False)
    coapplicant_income = Column(Float, default=0)
    monthly_expenses = Column(Float, nullable=False)
    other_emis = Column(Float, default=0)
    
    # Loan Details
    loan_amount = Column(Float, nullable=False)
    loan_amount_term = Column(Float, nullable=False)
    loan_purpose = Column(String(20), default="Personal")
    requested_interest_rate = Column(Float, nullable=True)
    
    # Credit Info
    credit_score = Column(Integer, default=650)
    credit_history = Column(Integer, nullable=False)
    no_of_credit_cards = Column(Integer, default=1)
    loan_default_history = Column(Integer, default=0)
    avg_payment_delay_days = Column(Float, default=0)
    
    # Assets
    has_vehicle = Column(Boolean, default=False)
    has_life_insurance = Column(Boolean, default=False)
    property_area = Column(String(20), nullable=False)
    
    # Banking
    bank_account_type = Column(String(20), default="Savings")
    bank_balance = Column(Float, default=50000)
    savings_score = Column(Float, default=10.0)
    
    # Collateral
    collateral_type = Column(String(20), default="None")
    collateral_value = Column(Float, default=0)
    
    # Geographic
    city_tier = Column(String(20), default="Tier-2")
    pincode = Column(String(6), default="110001")
    region_default_rate = Column(Float, default=5.0)
    
    # Prediction Results
    is_approved = Column(Boolean, nullable=True)
    confidence_score = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    prediction_details = Column(JSON, nullable=True)
    recommended_interest_rate = Column(Float, nullable=True)
    conditions = Column(JSON, nullable=True)
    
    # Model Information
    model_used = Column(String(50), nullable=True)
    model_performance = Column(JSON, nullable=True)

     # Metadata
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Decision Explanation Fields
    decision_summary = Column(Text, nullable=True)
    detailed_explanation = Column(Text, nullable=True)
    risk_explanation = Column(JSON, nullable=True)
    key_factors = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    next_steps = Column(JSON, nullable=True)
    explanation_metadata = Column(JSON, nullable=True)
    plain_text_summary = Column(Text, nullable=True)