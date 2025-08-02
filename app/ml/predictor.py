import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any, List
from .data_preprocessor import DataPreprocessor

class LoanPredictor:
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        self.model = None
        self.preprocessor = DataPreprocessor()
        
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
            # Create DataFrame from input data
            df = pd.DataFrame([loan_data])
            
            # Remove string columns and fields that shouldn't be processed 
            columns_to_remove = ['name', 'email', 'preferred_model']
            for col in columns_to_remove:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Apply the SAME feature engineering as training
            df = self._apply_training_feature_engineering(df)
            
            # Handle categorical variables with loaded encoders
            df = self._encode_categorical_features(df)
            
            # Apply weights if provided
            if weights:
                df = self._apply_weights(df, weights)
            
            # Handle missing values ONLY for numerical columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
            
            # Replace any remaining NaN with 0 for categorical columns
            df = df.fillna(0)
            
            # Scale features using loaded scaler
            X_scaled = self.preprocessor.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Calculate scores
            confidence_score = max(prediction_proba)
            risk_score = self._calculate_risk_score(loan_data, prediction_proba)
            
            # Generate detailed analysis
            feature_importance = self._get_feature_importance(X_scaled)
            conditions = self._generate_conditions(loan_data, prediction, risk_score)
            recommended_rate = self._calculate_recommended_rate(loan_data, risk_score)
            
            return {
                'approved': bool(prediction),
                'confidence_score': float(confidence_score),
                'risk_score': float(risk_score),
                'prediction_details': {
                    'feature_importance': feature_importance,
                    'risk_factors': self._identify_risk_factors(loan_data),
                    'positive_factors': self._identify_positive_factors(loan_data)
                },
                'recommended_interest_rate': recommended_rate,
                'conditions': conditions
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise ValueError(f"Error making prediction: {str(e)}")
    
    def _apply_training_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply EXACTLY the same feature engineering as in train_models.py"""
        
        # Income ratios
        df['total_income'] = df['applicant_income'] + df['coapplicant_income']
        df['loan_to_income_ratio'] = df['loan_amount'] / df['total_income']
        df['monthly_income'] = df['total_income'] / 12
        df['monthly_emi_estimated'] = df['loan_amount'] / df['loan_amount_term']
        df['emi_to_income_ratio'] = df['monthly_emi_estimated'] / df['monthly_income']
        
        # Expense ratios
        df['expense_to_income_ratio'] = df['monthly_expenses'] / df['monthly_income']
        df['total_obligations'] = df['monthly_expenses'] + df['other_emis']
        df['obligation_to_income_ratio'] = df['total_obligations'] / df['monthly_income']
        
        # Credit utilization and risk factors
        df['credit_utilization_estimated'] = df['no_of_credit_cards'] * 50000 * 0.3
        df['credit_age_score'] = df['age'] * 10 + df['credit_score']
        df['employment_stability_score'] = df['years_in_current_job'] * 10
        
        # Asset ratios
        df['collateral_to_loan_ratio'] = np.where(
            df['loan_amount'] > 0,
            df['collateral_value'] / df['loan_amount'],
            0
        )
        
        df['savings_months'] = np.where(
            df['monthly_expenses'] > 0,
            df['bank_balance'] / df['monthly_expenses'],
            0
        )
        
        # Risk composite scores
        df['financial_stability_score'] = (
            (df['total_income'] / 100000) +
            (df['years_in_current_job'] * 2) +
            (df['credit_score'] / 100) +
            (5 - df['loan_default_history']) +
            (df['has_life_insurance'].astype(int) * 2)
        )
        
        df['risk_factor_count'] = (
            (df['loan_default_history'] > 0).astype(int) +
            (df['avg_payment_delay_days'] > 30).astype(int) +
            (df['credit_score'] < 650).astype(int) +
            (df['obligation_to_income_ratio'] > 0.6).astype(int) +
            (df['region_default_rate'] > 8).astype(int)
        )
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using loaded encoders"""
        
        categorical_features = [
            'gender', 'married', 'education', 'self_employed', 'employment_type',
            'employer_category', 'industry', 'loan_purpose', 'property_area',
            'bank_account_type', 'collateral_type', 'city_tier'
        ]
        
        # Binary encoding for Yes/No fields
        binary_fields = {
            'married': {'Yes': 1, 'No': 0},
            'self_employed': {'Yes': 1, 'No': 0},
            'gender': {'Male': 1, 'Female': 0}
        }
        
        for field, mapping in binary_fields.items():
            if field in df.columns:
                df[field] = df[field].map(mapping)
        
        # Label encoding for multi-category fields using loaded encoders
        for feature in categorical_features:
            if feature in df.columns and feature not in binary_fields:
                if feature in self.preprocessor.label_encoders:
                    try:
                        df[feature] = self.preprocessor.label_encoders[feature].transform(df[feature].astype(str))
                    except ValueError:
                        # Handle unseen categories by using the most frequent category (0)
                        print(f"Warning: Unseen category in {feature}, using fallback value")
                        df[feature] = 0
                else:
                    print(f"Warning: No encoder found for {feature}")
                    df[feature] = 0
        
        return df
    
    def _apply_weights(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Apply field weights to features"""
        for column in df.columns:
            if column in weights:
                df[column] = df[column] * weights[column]
        return df
    
    def _calculate_risk_score(self, loan_data: Dict[str, Any], prediction_proba: np.ndarray) -> float:
        """Calculate risk score based on various factors"""
        
        # Base risk from model probability
        base_risk = (1 - prediction_proba[1]) * 100 if len(prediction_proba) > 1 else 50
        
        # Additional risk factors
        risk_adjustments = 0
        
        # Credit score risk
        credit_score = loan_data.get('credit_score', 650)
        if credit_score < 600:
            risk_adjustments += 20
        elif credit_score < 700:
            risk_adjustments += 10
        
        # Default history risk
        if loan_data.get('loan_default_history', 0) > 0:
            risk_adjustments += 15
        
        # High debt-to-income ratio
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        if total_income > 0:
            debt_ratio = (loan_data.get('other_emis', 0) * 12) / total_income
            if debt_ratio > 0.4:
                risk_adjustments += 10
        
        # Regional risk
        if loan_data.get('region_default_rate', 5) > 10:
            risk_adjustments += 5
        
        return min(100, max(0, base_risk + risk_adjustments))
    
    def _get_feature_importance(self, processed_data: np.ndarray) -> Dict[str, float]:
        """Get feature importance scores"""
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            feature_names = self.preprocessor.feature_names[:len(importance_scores)]
            
            return dict(zip(feature_names, importance_scores.tolist()))
        
        return {}
    
    def _identify_risk_factors(self, loan_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors in the application"""
        
        risk_factors = []
        
        if loan_data.get('credit_score', 650) < 600:
            risk_factors.append("Low credit score")
        
        if loan_data.get('loan_default_history', 0) > 0:
            risk_factors.append("Previous loan defaults")
        
        if loan_data.get('avg_payment_delay_days', 0) > 30:
            risk_factors.append("History of payment delays")
        
        total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
        if total_income > 0:
            loan_ratio = loan_data.get('loan_amount', 0) / total_income
            if loan_ratio > 5:
                risk_factors.append("High loan-to-income ratio")
        
        if loan_data.get('region_default_rate', 5) > 10:
            risk_factors.append("High regional default rate")
        
        return risk_factors
    
    def _identify_positive_factors(self, loan_data: Dict[str, Any]) -> List[str]:
        """Identify positive factors in the application"""
        
        positive_factors = []
        
        if loan_data.get('credit_score', 650) > 750:
            positive_factors.append("Excellent credit score")
        
        if loan_data.get('years_in_current_job', 0) > 5:
            positive_factors.append("Stable employment history")
        
        if loan_data.get('collateral_value', 0) > loan_data.get('loan_amount', 0):
            positive_factors.append("Adequate collateral coverage")
        
        if loan_data.get('savings_score', 10) > 20:
            positive_factors.append("Good savings habit")
        
        if loan_data.get('has_life_insurance', False):
            positive_factors.append("Life insurance coverage")
        
        return positive_factors
    
    def _generate_conditions(self, 
                           loan_data: Dict[str, Any], 
                           prediction: bool, 
                           risk_score: float) -> List[str]:
        """Generate loan conditions based on risk assessment"""
        
        conditions = []
        
        if risk_score > 70:
            conditions.append("Higher interest rate due to elevated risk")
        
        if loan_data.get('collateral_value', 0) == 0 and risk_score > 50:
            conditions.append("Collateral required")
        
        if loan_data.get('coapplicant_income', 0) == 0 and risk_score > 60:
            conditions.append("Co-applicant recommended")
        
        if prediction and risk_score > 40:
            conditions.append("Income verification required")
            conditions.append("Credit bureau verification required")
        
        return conditions
    
    def _calculate_recommended_rate(self, loan_data: Dict[str, Any], risk_score: float) -> float:
        """Calculate recommended interest rate"""
        
        base_rate = loan_data.get('requested_interest_rate', 12.0)
        
        # Risk-based pricing
        if risk_score > 70:
            base_rate += 3.0
        elif risk_score > 50:
            base_rate += 1.5
        elif risk_score > 30:
            base_rate += 0.5
        
        # Credit score adjustment
        credit_score = loan_data.get('credit_score', 650)
        if credit_score > 800:
            base_rate -= 1.0
        elif credit_score > 750:
            base_rate -= 0.5
        elif credit_score < 600:
            base_rate += 1.0
        
        return round(max(5.0, min(30.0, base_rate)), 2)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        self.model = joblib.load(model_path)
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model:
            joblib.dump(self.model, model_path)