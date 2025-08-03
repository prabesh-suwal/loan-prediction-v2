# # app/ml/feature_validator.py
# import joblib
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple

# class FeatureValidator:
#     """Ensures feature consistency between training and prediction"""
    
#     def __init__(self, preprocessing_path: str):
#         self.preprocessing_components = joblib.load(preprocessing_path)
#         self.expected_features = self.preprocessing_components['feature_names']
#         self.scaler = self.preprocessing_components['scaler']
#         self.label_encoders = self.preprocessing_components['label_encoders']
    
#     def validate_and_align_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Validate and align features to match training data"""
        
#         # Apply EXACT same feature engineering as training
#         df_processed = self._apply_training_feature_engineering(df)
        
#         # Ensure all expected features are present
#         missing_features = set(self.expected_features) - set(df_processed.columns)
#         if missing_features:
#             # Add missing features with default values
#             for feature in missing_features:
#                 df_processed[feature] = 0
#                 print(f"Warning: Added missing feature {feature} with default value 0")
        
#         # Remove any extra features not in training
#         extra_features = set(df_processed.columns) - set(self.expected_features)
#         if extra_features:
#             df_processed = df_processed.drop(columns=list(extra_features))
#             print(f"Warning: Removed extra features: {extra_features}")
        
#         # Ensure feature order matches training
#         df_processed = df_processed[self.expected_features]
        
#         return df_processed
    
#     def _apply_training_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Apply EXACTLY the same feature engineering as in train_models.py"""
        
#         # Remove non-feature columns
#         columns_to_remove = ['name', 'email', 'preferred_model']
#         for col in columns_to_remove:
#             if col in df.columns:
#                 df = df.drop(columns=[col])
        
#         # Engineer features - MUST match training exactly
#         df['total_income'] = df['applicant_income'] + df['coapplicant_income']
#         df['loan_to_income_ratio'] = df['loan_amount'] / df['total_income']
#         df['monthly_income'] = df['total_income'] / 12
#         df['monthly_emi_estimated'] = df['loan_amount'] / df['loan_amount_term']
#         df['emi_to_income_ratio'] = df['monthly_emi_estimated'] / df['monthly_income']
        
#         # Expense ratios
#         df['expense_to_income_ratio'] = df['monthly_expenses'] / df['monthly_income']
#         df['total_obligations'] = df['monthly_expenses'] + df['other_emis']
#         df['obligation_to_income_ratio'] = df['total_obligations'] / df['monthly_income']
        
#         # Credit and risk features
#         df['credit_utilization_estimated'] = df['no_of_credit_cards'] * 50000 * 0.3
#         df['credit_age_score'] = df['age'] * 10 + df['credit_score']
#         df['employment_stability_score'] = df['years_in_current_job'] * 10
        
#         # Asset ratios
#         df['collateral_to_loan_ratio'] = np.where(
#             df['loan_amount'] > 0,
#             df['collateral_value'] / df['loan_amount'],
#             0
#         )
        
#         df['savings_months'] = np.where(
#             df['monthly_expenses'] > 0,
#             df['bank_balance'] / df['monthly_expenses'],
#             0
#         )
        
#         # Composite scores
#         df['financial_stability_score'] = (
#             (df['total_income'] / 100000) +
#             (df['years_in_current_job'] * 2) +
#             (df['credit_score'] / 100) +
#             (5 - df['loan_default_history']) +
#             (df['has_life_insurance'].astype(int) * 2)
#         )
        
#         df['risk_factor_count'] = (
#             (df['loan_default_history'] > 0).astype(int) +
#             (df['avg_payment_delay_days'] > 30).astype(int) +
#             (df['credit_score'] < 650).astype(int) +
#             (df['obligation_to_income_ratio'] > 0.6).astype(int) +
#             (df['region_default_rate'] > 8).astype(int)
#         )
        
#         # Encode categorical variables
#         df = self._encode_categorical_features(df)
        
#         # Handle missing values
#         numerical_columns = df.select_dtypes(include=[np.number]).columns
#         df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
#         df = df.fillna(0)
        
#         return df
    
#     def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Encode categorical features using training encoders"""
        
#         # Binary fields
#         binary_fields = {
#             'married': {'Yes': 1, 'No': 0},
#             'self_employed': {'Yes': 1, 'No': 0},
#             'gender': {'Male': 1, 'Female': 0}
#         }
        
#         for field, mapping in binary_fields.items():
#             if field in df.columns:
#                 df[field] = df[field].map(mapping)
        
#         # Multi-category fields with label encoders
#         for feature, encoder in self.label_encoders.items():
#             if feature in df.columns:
#                 try:
#                     df[feature] = encoder.transform(df[feature].astype(str))
#                 except ValueError:
#                     # Handle unseen categories
#                     df[feature] = 0
#                     print(f"Warning: Unseen category in {feature}, using fallback")
        
#         return df