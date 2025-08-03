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
        self.is_fitted = False
        
    def preprocess_loan_data(self, data: Dict[str, Any], weights: Dict[str, float] = None) -> np.ndarray:
        """Preprocess loan application data for ML model - MUST match training preprocessing"""
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Remove string columns that shouldn't be processed (name, email, preferred_model)
        columns_to_remove = ['name', 'email', 'preferred_model']
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
        df_encoded[numerical_columns] = df_encoded[numerical_columns].fillna(0)  # Use 0 instead of median for consistency
        
        # Replace any remaining NaN with 0 for categorical columns
        df_encoded = df_encoded.fillna(0)
        
        # Ensure we have the correct number of features
        expected_features = len(self.feature_names) if self.feature_names else df_encoded.shape[1]
        
        # Align features with training data
        if self.feature_names and len(self.feature_names) > 0:
            # Reorder columns to match training order
            missing_cols = set(self.feature_names) - set(df_encoded.columns)
            extra_cols = set(df_encoded.columns) - set(self.feature_names)
            
            # Add missing columns with default values
            for col in missing_cols:
                df_encoded[col] = 0
            
            # Remove extra columns
            for col in extra_cols:
                df_encoded = df_encoded.drop(columns=[col])
            
            # Reorder to match training order
            df_encoded = df_encoded[self.feature_names]
        
        # Scale features
        if self.is_fitted:
            # Transform using fitted scaler
            X_scaled = self.scaler.transform(df_encoded)
        else:
            # This should only happen during training
            X_scaled = self.scaler.fit_transform(df_encoded)
            self.is_fitted = True
        
        return X_scaled
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features - MUST exactly match train_models.py feature engineering"""
        
        # Income ratios - MATCH training exactly
        df['total_income'] = df['applicant_income'] + df['coapplicant_income']
        
        # Prevent division by zero
        df['loan_to_income_ratio'] = np.where(
            df['total_income'] > 0,
            df['loan_amount'] / df['total_income'],
            0
        )
        
        df['monthly_income'] = df['total_income'] / 12
        
        df['monthly_emi_estimated'] = np.where(
            df['loan_amount_term'] > 0,
            df['loan_amount'] / df['loan_amount_term'],
            0
        )
        
        df['emi_to_income_ratio'] = np.where(
            df['monthly_income'] > 0,
            df['monthly_emi_estimated'] / df['monthly_income'],
            0
        )
        
        # Expense ratios - MATCH training exactly
        df['expense_to_income_ratio'] = np.where(
            df['monthly_income'] > 0,
            df['monthly_expenses'] / df['monthly_income'],
            0
        )
        
        df['total_obligations'] = df['monthly_expenses'] + df['other_emis']
        
        df['obligation_to_income_ratio'] = np.where(
            df['monthly_income'] > 0,
            df['total_obligations'] / df['monthly_income'],
            0
        )
        
        # Credit utilization and risk factors - MATCH training exactly
        df['credit_utilization_estimated'] = df['no_of_credit_cards'] * 50000 * 0.3
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
        """Encode categorical features using loaded encoders"""
        
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
                df[field] = df[field].map(mapping).fillna(0)
        
        # Label encoding for multi-category fields using loaded encoders
        for feature in categorical_features:
            if feature in df.columns and feature not in binary_fields:
                if feature in self.label_encoders:
                    try:
                        # Get unique values and handle unseen categories
                        unique_vals = df[feature].astype(str).unique()
                        known_classes = self.label_encoders[feature].classes_
                        
                        # Map known values, unknown values get 0 (first class)
                        def safe_transform(val):
                            if str(val) in known_classes:
                                return self.label_encoders[feature].transform([str(val)])[0]
                            else:
                                return 0
                        
                        df[feature] = df[feature].astype(str).apply(safe_transform)
                        
                    except (ValueError, AttributeError) as e:
                        # Handle any encoding errors
                        print(f"Warning: Error encoding {feature}: {e}")
                        df[feature] = 0
                else:
                    # No encoder found, use default value
                    print(f"Warning: No encoder found for {feature}")
                    df[feature] = 0
        
        return df
    
    def _apply_weights(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Apply field weights to features"""
        for column in df.columns:
            if column in weights:
                df[column] = df[column] * weights[column]
        return df
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform data during training"""
        # Store feature names for later alignment
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data during prediction"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Align features with training data
        if self.feature_names:
            # Add missing columns
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            
            # Remove extra columns and reorder
            X = X[self.feature_names]
        
        # Scale features
        return self.scaler.transform(X)
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor components"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor components"""
        components = joblib.load(filepath)
        self.scaler = components['scaler']
        self.label_encoders = components['label_encoders']
        self.feature_names = components['feature_names']
        self.is_fitted = components.get('is_fitted', True)