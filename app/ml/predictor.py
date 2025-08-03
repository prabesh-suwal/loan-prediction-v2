import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any, List
from .data_preprocessor import DataPreprocessor
import logging


logger = logging.getLogger(__name__)

class LoanPredictor:
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.model_type = None
        
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.preprocessor.load_preprocessor(preprocessor_path)
    
    def predict(self, 
                loan_data: Dict[str, Any], 
                weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Make loan approval prediction with proper feature alignment"""
        
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            # Log input data for debugging
            logger.info(f"Making prediction with model type: {self.model_type}")
            
            # Preprocess the data using the same pipeline as training
            processed_data = self.preprocessor.preprocess_loan_data(loan_data, weights)
            
            # Validate processed data shape
            if len(processed_data.shape) == 1:
                processed_data = processed_data.reshape(1, -1)
            
            logger.info(f"Processed data shape: {processed_data.shape}")
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(processed_data)[0]
                confidence_score = max(prediction_proba)
                approval_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            else:
                # For models without predict_proba (like SVM without probability=True)
                confidence_score = 0.8 if prediction else 0.7
                approval_probability = 0.8 if prediction else 0.2
            
            # Calculate comprehensive risk score
            risk_score = self._calculate_risk_score(loan_data, approval_probability)
            
            # Generate detailed analysis
            feature_analysis = self._analyze_features(loan_data, processed_data)
            conditions = self._generate_conditions(loan_data, prediction, risk_score)
            recommended_rate = self._calculate_recommended_rate(loan_data, risk_score)
            
            # Create comprehensive response
            result = {
                'approved': bool(prediction),
                'confidence_score': float(confidence_score),
                'risk_score': float(risk_score),
                'approval_probability': float(approval_probability),
                'prediction_details': {
                    'feature_analysis': feature_analysis,
                    'risk_factors': self._identify_risk_factors(loan_data),
                    'positive_factors': self._identify_positive_factors(loan_data),
                    'score_breakdown': self._get_score_breakdown(loan_data)
                },
                'recommended_interest_rate': recommended_rate,
                'conditions': conditions
            }
            
            logger.info(f"Prediction result: {result['approved']} with confidence {result['confidence_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f" ❌ Prediction error: {e}")
            # Return a safe fallback response
            return {
                'approved': False,
                'confidence_score': 0.0,
                'risk_score': 100.0,
                'prediction_details': {
                    'error': f"Prediction failed: {str(e)}",
                    'risk_factors': ['System error during prediction'],
                    'positive_factors': [],
                    'feature_analysis': {}
                },
                'recommended_interest_rate': 15.0,
                'conditions': ['Manual review required due to system error']
            }
    
    def _calculate_risk_score(self, loan_data: Dict[str, Any], approval_probability: float) -> float:
        """Calculate comprehensive risk score based on multiple factors"""
        
        # Base risk from model probability (inverted)
        base_risk = (1 - approval_probability) * 50
        
        # Credit score risk component
        credit_score = loan_data.get('credit_score', 650)
        if credit_score < 500:
            credit_risk = 30
        elif credit_score < 600:
            credit_risk = 20
        elif credit_score < 700:
            credit_risk = 10
        elif credit_score < 750:
            credit_risk = 5
        else:
            credit_risk = 0
        
        # Income stability risk
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        
        if total_income > 0:
            loan_to_income = loan_amount / total_income
            if loan_to_income > 10:
                income_risk = 15
            elif loan_to_income > 6:
                income_risk = 10
            elif loan_to_income > 4:
                income_risk = 5
            else:
                income_risk = 0
        else:
            income_risk = 25
        
        # Employment stability risk
        years_in_job = loan_data.get('years_in_current_job', 0)
        if years_in_job < 1:
            employment_risk = 10
        elif years_in_job < 2:
            employment_risk = 5
        else:
            employment_risk = 0
        
        # Default history risk
        default_history = loan_data.get('loan_default_history', 0)
        default_risk = min(default_history * 8, 20)
        
        # Regional risk
        regional_rate = loan_data.get('region_default_rate', 5)
        if regional_rate > 10:
            regional_risk = 8
        elif regional_rate > 7:
            regional_risk = 5
        elif regional_rate > 5:
            regional_risk = 2
        else:
            regional_risk = 0
        
        # Combine all risk factors
        total_risk = base_risk + credit_risk + income_risk + employment_risk + default_risk + regional_risk
        
        # Normalize to 0-100 scale
        return min(100, max(0, total_risk))
    
    def _analyze_features(self, loan_data: Dict[str, Any], processed_data: np.ndarray) -> Dict[str, Any]:
        """Analyze feature contributions to the prediction"""
        
        analysis = {}
        
        # Calculate key financial ratios
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        monthly_expenses = loan_data.get('monthly_expenses', 0)
        
        if total_income > 0:
            analysis['loan_to_income_ratio'] = loan_amount / total_income
            analysis['expense_to_income_ratio'] = (monthly_expenses * 12) / total_income
            analysis['monthly_income'] = total_income / 12
        
        # Employment stability score
        analysis['employment_stability'] = loan_data.get('years_in_current_job', 0) * 2
        
        # Credit health score
        credit_score = loan_data.get('credit_score', 650)
        analysis['credit_health'] = min(100, (credit_score - 300) / 5.5)
        
        # Asset coverage
        collateral_value = loan_data.get('collateral_value', 0)
        if loan_amount > 0:
            analysis['collateral_coverage'] = min(100, (collateral_value / loan_amount) * 100)
        else:
            analysis['collateral_coverage'] = 0
        
        return analysis
    
    def _get_score_breakdown(self, loan_data: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed score breakdown for transparency"""
        
        breakdown = {}
        
        # Credit score component (40% weight)
        credit_score = loan_data.get('credit_score', 650)
        breakdown['credit_score_component'] = min(40, (credit_score - 300) / 550 * 40)
        
        # Income stability component (25% weight)
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        if total_income > 0 and loan_amount > 0:
            income_ratio = total_income / loan_amount
            breakdown['income_stability_component'] = min(25, income_ratio * 5)
        else:
            breakdown['income_stability_component'] = 0
        
        # Employment history component (20% weight)
        years_in_job = loan_data.get('years_in_current_job', 0)
        breakdown['employment_component'] = min(20, years_in_job * 4)
        
        # Credit history component (15% weight)
        credit_history = loan_data.get('credit_history', 0)
        default_history = loan_data.get('loan_default_history', 0)
        breakdown['credit_history_component'] = max(0, 15 - (default_history * 5))
        
        # Calculate total
        breakdown['total_score'] = sum(breakdown.values())
        
        return breakdown
    
    def _identify_risk_factors(self, loan_data: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors in the application"""
        
        risk_factors = []
        
        # Credit score risks
        credit_score = loan_data.get('credit_score', 650)
        if credit_score < 600:
            risk_factors.append(f"Low credit score ({credit_score})")
        
        # Default history risks
        default_history = loan_data.get('loan_default_history', 0)
        if default_history > 0:
            risk_factors.append(f"Previous loan defaults ({default_history})")
        
        # Payment delay risks
        avg_delay = loan_data.get('avg_payment_delay_days', 0)
        if avg_delay > 30:
            risk_factors.append(f"History of payment delays (avg {avg_delay:.0f} days)")
        
        # Income vs loan amount risks
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        if total_income > 0:
            ratio = loan_amount / total_income
            if ratio > 8:
                risk_factors.append(f"Very high loan-to-income ratio ({ratio:.1f}x)")
            elif ratio > 5:
                risk_factors.append(f"High loan-to-income ratio ({ratio:.1f}x)")
        
        # Employment stability risks
        years_in_job = loan_data.get('years_in_current_job', 0)
        if years_in_job < 1:
            risk_factors.append("Limited employment history")
        
        # Regional risks
        regional_rate = loan_data.get('region_default_rate', 5)
        if regional_rate > 10:
            risk_factors.append(f"High regional default rate ({regional_rate:.1f}%)")
        
        # Age-related risks
        age = loan_data.get('age', 30)
        if age < 25:
            risk_factors.append("Young applicant with limited credit history")
        elif age > 60:
            risk_factors.append("Near retirement age")
        
        return risk_factors
    
    def _identify_positive_factors(self, loan_data: Dict[str, Any]) -> List[str]:
        """Identify positive factors that support approval"""
        
        positive_factors = []
        
        # Credit score positives
        credit_score = loan_data.get('credit_score', 650)
        if credit_score > 800:
            positive_factors.append("Excellent credit score")
        elif credit_score > 750:
            positive_factors.append("Very good credit score")
        elif credit_score > 700:
            positive_factors.append("Good credit score")
        
        # Employment stability
        years_in_job = loan_data.get('years_in_current_job', 0)
        if years_in_job > 5:
            positive_factors.append("Long-term stable employment")
        elif years_in_job > 3:
            positive_factors.append("Stable employment history")
        
        # Income factors
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        if total_income > 100000:
            positive_factors.append("High income level")
        elif total_income > 60000:
            positive_factors.append("Good income level")
        
        # Collateral coverage
        collateral_value = loan_data.get('collateral_value', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        if collateral_value > loan_amount * 1.5:
            positive_factors.append("Excellent collateral coverage")
        elif collateral_value > loan_amount:
            positive_factors.append("Good collateral coverage")
        
        # Assets and insurance
        if loan_data.get('has_life_insurance', False):
            positive_factors.append("Life insurance coverage")
        
        if loan_data.get('has_vehicle', False):
            positive_factors.append("Vehicle ownership")
        
        # Savings
        savings_score = loan_data.get('savings_score', 10)
        if savings_score > 25:
            positive_factors.append("Excellent savings habit")
        elif savings_score > 15:
            positive_factors.append("Good savings pattern")
        
        # Education
        if loan_data.get('education') == 'Graduate':
            positive_factors.append("Graduate education")
        
        # Employment type
        employment_type = loan_data.get('employment_type', '')
        if employment_type == 'Government':
            positive_factors.append("Government employment")
        
        return positive_factors
    
    def _generate_conditions(self, 
                           loan_data: Dict[str, Any], 
                           prediction: bool, 
                           risk_score: float) -> List[str]:
        """Generate specific loan conditions based on risk assessment"""
        
        conditions = []
        
        # Risk-based conditions
        if risk_score > 80:
            conditions.append("Requires senior management approval")
            conditions.append("Enhanced documentation required")
        elif risk_score > 60:
            conditions.append("Additional income verification required")
            conditions.append("Credit bureau re-verification needed")
        elif risk_score > 40:
            conditions.append("Standard income verification required")
        
        # Collateral conditions
        collateral_value = loan_data.get('collateral_value', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        if collateral_value < loan_amount * 0.8 and risk_score > 50:
            conditions.append("Additional collateral or guarantor required")
        
        # Co-applicant conditions
        if loan_data.get('coapplicant_income', 0) == 0 and risk_score > 50:
            conditions.append("Co-applicant with stable income recommended")
        
        # Credit score conditions
        credit_score = loan_data.get('credit_score', 650)
        if credit_score < 650:
            conditions.append("Credit improvement plan required")
        
        # Insurance conditions
        if not loan_data.get('has_life_insurance', False) and loan_amount > 500000:
            conditions.append("Life insurance policy required")
        
        # Tenure conditions
        loan_tenure = loan_data.get('loan_amount_term', 12)
        if loan_tenure > 240:  # More than 20 years
            conditions.append("Long tenure requires additional scrutiny")
        
        return conditions
    
    def _calculate_recommended_rate(self, loan_data: Dict[str, Any], risk_score: float) -> float:
        """Calculate risk-adjusted interest rate recommendation"""
        
        # Base rate (market rate)
        base_rate = 12
        
        # Risk-based pricing adjustment
        if risk_score > 80:
            risk_adjustment = 4.0
        elif risk_score > 60:
            risk_adjustment = 2.5
        elif risk_score > 40:
            risk_adjustment = 1.0
        elif risk_score > 20:
            risk_adjustment = 0.5
        else:
            risk_adjustment = -0.5  # Discount for very low risk
        
        # Credit score adjustment
        credit_score = loan_data.get('credit_score', 650)
        if credit_score > 800:
            credit_adjustment = -1.0
        elif credit_score > 750:
            credit_adjustment = -0.5
        elif credit_score < 600:
            credit_adjustment = 1.5
        elif credit_score < 650:
            credit_adjustment = 0.5
        else:
            credit_adjustment = 0.0
        
        # Loan amount adjustment (larger loans get better rates)
        loan_amount = loan_data.get('loan_amount', 0)
        if loan_amount > 1000000:
            amount_adjustment = -0.5
        elif loan_amount > 500000:
            amount_adjustment = -0.25
        elif loan_amount < 100000:
            amount_adjustment = 0.5
        else:
            amount_adjustment = 0.0
        
        # Calculate final rate
        final_rate = base_rate + risk_adjustment + credit_adjustment + amount_adjustment
        
        # Ensure rate is within reasonable bounds
        return round(max(6.0, min(30.0, final_rate)), 2)
    
    def load_model(self, model_path: str):
        """Load trained model and detect model type"""
        self.model = joblib.load(model_path)
        
        # Detect model type for logging
        model_class = self.model.__class__.__name__
        self.model_type = model_class
        logger.info(f"Loaded model type: {model_class}")
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model:
            joblib.dump(self.model, model_path)
            logger.info(f"Saved model to {model_path}")
        else:
            raise ValueError("No model to save")