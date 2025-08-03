from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

# class PredictionResult(BaseModel):
#     approved: bool
#     confidence_score: float
#     risk_score: float
#     prediction_details: dict
#     recommended_interest_rate: Optional[float]
#     conditions: list[str]

# class LoanApplicationResponse(BaseModel):
#     id: int
#     status: str
#     prediction_result: Optional[PredictionResult]
#     created_at: datetime
#     updated_at: datetime

class RiskExplanation(BaseModel):
    risk_score: float
    risk_category: str
    risk_explanation: str
    risk_breakdown: Dict[str, Any]

class KeyFactors(BaseModel):
    positive_factors: List[str]
    risk_factors: List[str]
    factor_explanations: Dict[str, str]

class ExplanationMetadata(BaseModel):
    confidence_level: str
    risk_category: str
    decision_basis: str

class DecisionExplanation(BaseModel):
    decision_summary: str
    detailed_explanation: str
    risk_assessment: RiskExplanation
    key_factors: KeyFactors
    recommendations: List[str]
    next_steps: List[str]
    explanation_metadata: ExplanationMetadata

class PredictionResult(BaseModel):
    approved: bool
    confidence_score: float
    risk_score: float
    prediction_details: dict
    recommended_interest_rate: Optional[float]
    conditions: List[str]
    # NEW: Add explanation
    explanation: Optional[DecisionExplanation] = None

class LoanApplicationResponse(BaseModel):
    id: int
    status: str
    prediction_result: Optional[PredictionResult]
    created_at: datetime
    updated_at: datetime
    model_used: Optional[str] = None
    
    # NEW: Direct explanation fields for easy access
    decision_summary: Optional[str] = None
    plain_text_summary: Optional[str] = None

class LoanApplicationDetailedResponse(LoanApplicationResponse):
    """Extended response with full explanation details"""
    detailed_explanation: Optional[str] = None
    risk_explanation: Optional[Dict[str, Any]] = None
    key_factors: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    explanation_metadata: Optional[Dict[str, Any]] = None
    
class PaginationMetadata(BaseModel):
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    total_count: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")
    next_page: Optional[int] = Field(None, description="Next page number if available")
    previous_page: Optional[int] = Field(None, description="Previous page number if available")

class LoanApplicationSummary(BaseModel):
    """Summary statistics for the current page/filter"""
    total_applications: int = Field(..., description="Total applications matching filters")
    approved_count: int = Field(..., description="Number of approved applications")
    rejected_count: int = Field(..., description="Number of rejected applications") 
    pending_count: int = Field(..., description="Number of pending applications")

class PaginatedLoanResponse(BaseModel):
    """Paginated response for loan applications list"""
    data: List[LoanApplicationResponse] = Field(..., description="List of loan applications")
    pagination: PaginationMetadata = Field(..., description="Pagination information")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    summary: LoanApplicationSummary = Field(..., description="Summary statistics")

# Additional filter schemas for better API documentation
class LoanApplicationFilters(BaseModel):
    """Available filters for loan applications"""
    name: Optional[str] = Field(None, description="Filter by applicant name")
    email: Optional[str] = Field(None, description="Filter by email")
    status: Optional[str] = Field(None, description="Filter by status")
    approved: Optional[bool] = Field(None, description="Filter by approval status")
    risk_category: Optional[str] = Field(None, description="Filter by risk category")
    credit_score_min: Optional[int] = Field(None, ge=300, le=850)
    credit_score_max: Optional[int] = Field(None, ge=300, le=850)
    loan_amount_min: Optional[float] = Field(None, ge=0)
    loan_amount_max: Optional[float] = Field(None, ge=0)
    sort_by: Optional[str] = Field("created_at", description="Sort field")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$")

# Quick response schemas for mobile/lightweight clients
class LoanApplicationQuickResponse(BaseModel):
    """Minimal response for quick listing"""
    id: int
    name: str
    loan_amount: float
    status: str
    approved: Optional[bool]
    confidence_score: Optional[float]
    risk_score: Optional[float]
    created_at: datetime

class PaginatedQuickLoanResponse(BaseModel):
    """Paginated response with minimal loan data"""
    data: List[LoanApplicationQuickResponse]
    pagination: PaginationMetadata
    summary: LoanApplicationSummary