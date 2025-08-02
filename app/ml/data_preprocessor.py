import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Any
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def preprocess_loan_data(self, data: Dict[str, Any], weights: Dict[str, float] = None) -> np.ndarray:
        """Preprocess loan application data for ML model - MUST match training preprocessing"""
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Remove string columns that shouldn't be processed (name, email)
        columns_to_remove = ['name', 'email']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Feature engineering - EXACTLY match train_models.py
        df = self._engineer_features(df)
        
        # Handle categorical variables - EXACTLY match train_models.py
        df_encoded = self._encode_categorical_features(df)
        
        # Apply weights if provided
        if weights:
            df_encoded = self._apply_weights(df_encoded, weights)
        
        # Handle missing values ONLY for numerical columns
        numerical_columns = df_encoded.select_dtypes(include=[np.number]).columns
        df_encoded[numerical_columns] = df_encoded[numerical_columns].fillna(df_encoded[numerical_columns].median())
        
        # Replace any remaining NaN with 0 for categorical columns
        df_encoded = df_encoded.fillna(0)
        
        # Scale numerical features
        df_scaled = self._scale_features(df_encoded)
        
        return df_scaled
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features - MUST exactly match train_models.py feature engineering"""
        
        # Income ratios - MATCH training exactly
        df['total_income'] = df['applicant_income'] + df['coapplicant_income']
        df['loan_to_income_ratio'] = df['loan_amount'] / df['total_income']
        df['monthly_income'] = df['total_income'] / 12
        df['monthly_emi_estimated'] = df['loan_amount'] / df['loan_amount_term']
        df['emi_to_income_ratio'] = df['monthly_emi_estimated'] / df['monthly_income']
        
        # Expense ratios - MATCH training exactly
        df['expense_to_income_ratio'] = df['monthly_expenses'] / df['monthly_income']
        df['total_obligations'] = df['monthly_expenses'] + df['other_emis']
        df['obligation_to_income_ratio'] = df['total_obligations'] / df['monthly_income']
        
        # Credit utilization and risk factors - MATCH training exactly
        df['credit_utilization_estimated'] = df['no_of_credit_cards'] * 50000 * 0.3  # Estimated usage
        df['credit_age_score'] = df['age'] * 10 + df['credit_score']
        df['employment_stability_score'] = df['years_in_current_job'] * 10
        
        # Asset ratios - MATCH training exactly
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
        
        # Risk composite scores - MATCH training exactly
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
        """Encode categorical features - MUST exactly match train_models.py"""
        
        categorical_features = [
            'gender', 'married', 'education', 'self_employed', 'employment_type',
            'employer_category', 'industry', 'loan_purpose', 'property_area',
            'bank_account_type', 'collateral_type', 'city_tier'
        ]
        
        # Binary encoding for Yes/No fields - MATCH training exactly
        binary_fields = {
            'married': {'Yes': 1, 'No': 0},
            'self_employed': {'Yes': 1, 'No': 0},
            'gender': {'Male': 1, 'Female': 0}
        }
        
        for field, mapping in binary_fields.items():
            if field in df.columns:
                df[field] = df[field].map(mapping)
        
        # Label encoding for multi-category fields - MATCH training exactly
        for feature in categorical_features:
            if feature in df.columns and feature not in binary_fields:
                if feature not in self.label_encoders:
                    # This should not happen in prediction - encoders should be loaded
                    print(f"Warning: No encoder found for {feature}")
                    continue
                else:
                    # Handle unseen categories gracefully
                    try:
                        df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
                    except ValueError as e:
                        # Handle unseen category by assigning most frequent category
                        print(f"Warning: Unseen category in {feature}, using fallback")
                        df[feature] = 0  # Fallback to first category
        
        return df
    
    def _apply_weights(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Apply field weights to features"""
        
        for column in df.columns:
            if column in weights:
                df[column] = df[column] * weights[column]
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale numerical features"""
        
        # Select numerical columns
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if hasattr(self.scaler, 'n_features_in_'):
            # Transform using fitted scaler
            return self.scaler.transform(df[numerical_features])
        else:
            # Fit and transform (only during training)
            return self.scaler.fit_transform(df[numerical_features])
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor components"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor components"""
        components = joblib.load(filepath)
        self.scaler = components['scaler']
        self.label_encoders = components['label_encoders']
        self.feature_names = components['feature_names']