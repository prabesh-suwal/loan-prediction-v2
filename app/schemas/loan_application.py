from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class LoanApplicationCreate(BaseModel):

    preferred_model: Optional[str] = Field(default=None, 
        description="Preferred ML model type for prediction. If not specified, default model will be used.")

    name: str = Field(..., min_length=2, max_length=100, description="Full name of the applicant")
    email: str = Field(..., min_length=2, max_length=100, description="Email address of the applicant")
    
    gender: str = Field(..., pattern="^(Male|Female)$")
    married: str = Field(..., pattern="^(Yes|No)$")
    dependents: int = Field(..., ge=0, le=10)
    education: str = Field(..., pattern="^(Graduate|Not Graduate)$")
    age: int = Field(..., ge=18, le=80, description="Applicant age")
    children: int = Field(default=0, ge=0, le=10, description="Number of children")
    spouse_employed: bool = Field(default=False, description="Is spouse employed")
    
    # Employment & Stability
    self_employed: str = Field(..., pattern="^(Yes|No)$")
    employment_type: str = Field(default="Salaried", pattern="^(Salaried|Self-Employed|Government|Freelancer|Business Owner)$")
    years_in_current_job: float = Field(default=2.0, ge=0, le=50, description="Years in current job")
    employer_category: str = Field(default="B", pattern="^(A|B|C|SME|MNC)$", description="Employer rating")
    industry: str = Field(default="Others", pattern="^(Finance|IT|Healthcare|Retail|Manufacturing|Government|Education|Others)$")
    
    # Income & Expenses
    applicant_income: float = Field(..., gt=0)
    coapplicant_income: float = Field(default=0, ge=0)
    monthly_expenses: float = Field(..., ge=0, description="Total monthly expenses")
    other_emis: float = Field(default=0, ge=0, description="Existing EMI obligations")
    
    # Loan Details
    loan_amount: float = Field(..., gt=0)
    loan_amount_term: float = Field(..., gt=0)
    loan_purpose: str = Field(default="Personal", pattern="^(Home|Personal|Education|Business|Vehicle|Medical|Others)$")
    requested_interest_rate: Optional[float] = Field(default=None, ge=5.0, le=30.0)
    
    # Credit History & Behavior
    credit_score: int = Field(default=650, ge=300, le=850, description="Credit bureau score")
    credit_history: int = Field(..., ge=0, le=1, description="1=Good, 0=Poor")
    no_of_credit_cards: int = Field(default=1, ge=0, le=20)
    loan_default_history: int = Field(default=0, ge=0, le=10, description="Number of past defaults")
    avg_payment_delay_days: float = Field(default=0, ge=0, le=365, description="Average payment delay")
    
    # Assets & Lifestyle
    has_vehicle: bool = Field(default=False)
    has_life_insurance: bool = Field(default=False)
    property_area: str = Field(..., pattern="^(Urban|Semiurban|Rural)$")
    
    # Banking Info
    bank_account_type: str = Field(default="Savings", pattern="^(Basic|Savings|Premium|Current)$")
    bank_balance: float = Field(default=50000, ge=0, description="Current bank balance")
    savings_score: float = Field(default=10.0, ge=0, le=100, description="% of income saved monthly")
    
    # Collateral
    collateral_type: str = Field(default="None", pattern="^(Property|Vehicle|Fixed Deposit|None)$")
    collateral_value: float = Field(default=0, ge=0, description="Value of collateral")
    
    # Geographic Info
    city_tier: str = Field(default="Tier-2", pattern="^(Tier-1|Tier-2|Tier-3)$")
    pincode: str = Field(default="110001", pattern="^[0-9]{6}$", description="6-digit pincode")
    region_default_rate: float = Field(default=5.0, ge=0, le=100, description="Regional default rate %")

class PredictionResult(BaseModel):
    approved: bool
    confidence_score: float
    risk_score: float
    prediction_details: dict
    recommended_interest_rate: Optional[float]
    conditions: list[str]

class LoanApplicationResponse(BaseModel):
    id: int
    status: str
    prediction_result: Optional[PredictionResult]
    created_at: datetime
    updated_at: datetime
