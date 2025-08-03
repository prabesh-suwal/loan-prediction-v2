# scripts/enhanced_train_models.py - ROBUST TRAINING SCRIPT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustModelTrainer:
    def __init__(self, models_to_train=None, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_performances = {}
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.validation_results = {}
        
        # Initialize available models with robust configurations
        self._initialize_models(models_to_train)
        
        # Define comprehensive parameter grids
        self._define_parameter_grids()
        
    def _initialize_models(self, models_to_train=None):
        """Initialize ML models with robust default configurations"""
        
        available_models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=2000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            available_models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            available_models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                verbose=-1
            )
        
        # Filter models if specified
        if models_to_train:
            self.models = {k: v for k, v in available_models.items() if k in models_to_train}
        else:
            self.models = available_models
            
        logger.info(f"Initialized models: {list(self.models.keys())}")
    
    def _define_parameter_grids(self):
        """Define parameter grids for hyperparameter tuning"""
        
        self.param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.param_grids['xgboost'] = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0.1, 1]
            }
        
        if LIGHTGBM_AVAILABLE:
            self.param_grids['lightgbm'] = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0.1, 1]
            }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'loan_status') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Robust data preparation with comprehensive preprocessing"""
        
        logger.info("🔄 Starting robust data preparation...")
        
        # Validate input data
        if df.empty:
            raise ValueError("Empty dataframe provided")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Original data shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Remove problematic columns
        columns_to_remove = ['name', 'email', 'preferred_model']
        for col in columns_to_remove:
            if col in X.columns:
                X = X.drop(columns=[col])
                logger.info(f"Removed column: {col}")
        
        # Convert target to binary
        if y.dtype == 'object':
            y = (y == 'Y').astype(int)
        
        # Handle missing values BEFORE feature engineering
        X = self._handle_missing_values(X)
        
        # Feature engineering
        X = self._engineer_features(X)
        
        # Handle categorical variables
        X = self._encode_categorical_features(X)
        
        # Feature selection and validation
        X = self._validate_and_clean_features(X)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Final missing value check
        if X.isnull().sum().sum() > 0:
            logger.warning("Found remaining missing values, filling with 0")
            X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"✅ Data preparation completed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"📊 Final target distribution: {np.bincount(y)} (0: Reject, 1: Approve)")
        
        return X_scaled, y.values, self.feature_names
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value handling"""
        
        missing_summary = X.isnull().sum()
        if missing_summary.sum() > 0:
            logger.info(f"Found missing values in {missing_summary[missing_summary > 0].shape[0]} columns")
            
            for col in missing_summary[missing_summary > 0].index:
                if X[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_value = X[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    X[col] = X[col].fillna(fill_value)
                else:
                    # Fill numerical with median
                    X[col] = X[col].fillna(X[col].median())
                
                logger.info(f"Filled missing values in {col}")
        
        return X
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering with error handling"""
        
        logger.info("🔧 Engineering features...")
        
        try:
            # Income and financial ratios
            X['total_income'] = X['applicant_income'] + X['coapplicant_income']
            
            # Safe division operations
            X['loan_to_income_ratio'] = np.where(
                X['total_income'] > 0,
                X['loan_amount'] / X['total_income'],
                0
            )
            
            X['monthly_income'] = X['total_income'] / 12
            
            X['monthly_emi_estimated'] = np.where(
                X['loan_amount_term'] > 0,
                X['loan_amount'] / X['loan_amount_term'],
                0
            )
            
            X['emi_to_income_ratio'] = np.where(
                X['monthly_income'] > 0,
                X['monthly_emi_estimated'] / X['monthly_income'],
                0
            )
            
            # Expense and obligation ratios
            X['expense_to_income_ratio'] = np.where(
                X['monthly_income'] > 0,
                X['monthly_expenses'] / X['monthly_income'],
                0
            )
            
            X['total_obligations'] = X['monthly_expenses'] + X.get('other_emis', 0)
            
            X['obligation_to_income_ratio'] = np.where(
                X['monthly_income'] > 0,
                X['total_obligations'] / X['monthly_income'],
                0
            )
            
            # Credit and risk factors
            X['credit_utilization_estimated'] = X.get('no_of_credit_cards', 1) * 50000 * 0.3
            X['credit_age_score'] = X['age'] * 10 + X.get('credit_score', 650)
            X['employment_stability_score'] = X.get('years_in_current_job', 0) * 10
            
            # Asset ratios
            X['collateral_to_loan_ratio'] = np.where(
                X['loan_amount'] > 0,
                X.get('collateral_value', 0) / X['loan_amount'],
                0
            )
            
            X['savings_months'] = np.where(
                X['monthly_expenses'] > 0,
                X.get('bank_balance', 0) / X['monthly_expenses'],
                0
            )
            
            # Composite risk scores
            X['financial_stability_score'] = (
                (X['total_income'] / 100000) +
                (X.get('years_in_current_job', 0) * 2) +
                (X.get('credit_score', 650) / 100) +
                (5 - X.get('loan_default_history', 0)) +
                (X.get('has_life_insurance', False).astype(int) * 2)
            )
            
            X['risk_factor_count'] = (
                (X.get('loan_default_history', 0) > 0).astype(int) +
                (X.get('avg_payment_delay_days', 0) > 30).astype(int) +
                (X.get('credit_score', 650) < 650).astype(int) +
                (X['obligation_to_income_ratio'] > 0.6).astype(int) +
                (X.get('region_default_rate', 5) > 8).astype(int)
            )
            
            # Additional derived features
            X['income_per_dependent'] = np.where(
                X.get('dependents', 0) > 0,
                X['total_income'] / X.get('dependents', 1),
                X['total_income']
            )
            
            X['loan_amount_per_year_term'] = np.where(
                X['loan_amount_term'] > 0,
                X['loan_amount'] / (X['loan_amount_term'] / 12),
                0
            )
            
            # Replace any infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            logger.info(f"✅ Feature engineering completed: {X.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Robust categorical encoding with error handling"""
        
        logger.info("🏷️  Encoding categorical features...")
        
        # Define categorical features
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
            if field in X.columns:
                X[field] = X[field].map(mapping).fillna(0)
                logger.info(f"Binary encoded: {field}")
        
        # Label encoding for multi-category fields
        for feature in categorical_features:
            if feature in X.columns and feature not in binary_fields:
                try:
                    le = LabelEncoder()
                    # Convert to string and handle missing values
                    X[feature] = X[feature].astype(str).fillna('Unknown')
                    X[feature] = le.fit_transform(X[feature])
                    self.label_encoders[feature] = le
                    logger.info(f"Label encoded: {feature} ({len(le.classes_)} categories)")
                except Exception as e:
                    logger.warning(f"Error encoding {feature}: {e}")
                    X[feature] = 0
        
        return X
    
    def _validate_and_clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean features"""
        
        logger.info("🧹 Validating and cleaning features...")
        
        # Remove features with zero variance
        zero_var_cols = X.columns[X.var() == 0].tolist()
        if zero_var_cols:
            X = X.drop(columns=zero_var_cols)
            logger.info(f"Removed zero variance columns: {zero_var_cols}")
        
        # Remove highly correlated features
        correlation_matrix = X.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.95:
                    col_to_remove = correlation_matrix.columns[j]
                    high_corr_pairs.append((correlation_matrix.columns[i], col_to_remove))
        
        if high_corr_pairs:
            cols_to_remove = list(set([pair[1] for pair in high_corr_pairs]))
            X = X.drop(columns=cols_to_remove)
            logger.info(f"Removed highly correlated columns: {cols_to_remove}")
        
        # Cap extreme outliers
        for col in X.select_dtypes(include=[np.number]).columns:
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            X[col] = X[col].clip(lower=q01, upper=q99)
        
        return X
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Train all models with comprehensive evaluation"""
        
        logger.info(f"🤖 Training {len(self.models)} models...")
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"🔄 Training {model_name}...")
            
            try:
                start_time = datetime.now()
                
                # Train model
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Cross-validation
                cv_scores = self._cross_validate_model(model, X_train, y_train)
                
                # Feature importance
                feature_importance = self._get_feature_importance(model)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'training_time': training_time,
                    'feature_importance': feature_importance,
                    'cv_scores': cv_scores,
                    **metrics
                }
                
                # Store for saving
                self.trained_models[model_name] = model
                self.model_performances[model_name] = {
                    'accuracy': metrics['accuracy'],
                    'roc_auc': metrics['roc_auc'],
                    'cv_mean': cv_scores['mean']
                }
                
                logger.info(f"✅ {model_name} - ROC-AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Identify best model
        successful_models = [name for name in results.keys() if 'error' not in results[name]]
        if successful_models:
            best_model_name = max(successful_models, key=lambda name: results[name]['roc_auc'])
            
            results['training_summary'] = {
                'total_models': len(successful_models),
                'best_model': best_model_name,
                'best_score': results[best_model_name]['roc_auc'],
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            }
            
            logger.info(f"🏆 Best model: {best_model_name} with ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['avg_precision'] = np.mean(precision)
            metrics['avg_recall'] = np.mean(recall)
        else:
            metrics['roc_auc'] = 0.0
            metrics['avg_precision'] = 0.0
            metrics['avg_recall'] = 0.0
        
        return metrics
    
    def _cross_validate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform stratified cross-validation"""
        
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # ROC-AUC cross-validation
            roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            # Accuracy cross-validation
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            
            return {
                'roc_auc_scores': roc_scores.tolist(),
                'accuracy_scores': acc_scores.tolist(),
                'mean': roc_scores.mean(),
                'std': roc_scores.std(),
                'accuracy_mean': acc_scores.mean(),
                'accuracy_std': acc_scores.std()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {'mean': 0.0, 'std': 0.0, 'accuracy_mean': 0.0, 'accuracy_std': 0.0}
    
    def _get_feature_importance(self, model) -> Optional[Dict[str, float]]:
        """Extract feature importance from model"""
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_values = np.abs(model.coef_[0])
            else:
                return None
            
            # Create feature importance dictionary
            feature_names = self.feature_names[:len(importance_values)]
            return dict(zip(feature_names, importance_values.tolist()))
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, models_to_tune: List[str] = None) -> Dict[str, Any]:
        """Perform hyperparameter tuning with cross-validation"""
        
        if models_to_tune is None:
            models_to_tune = list(self.models.keys())
        
        logger.info(f"🎯 Starting hyperparameter tuning for: {models_to_tune}")
        
        tuning_results = {}
        
        for model_name in models_to_tune:
            if model_name not in self.models or model_name not in self.param_grids:
                continue
            
            logger.info(f"🔧 Tuning {model_name}...")
            
            try:
                model = self.models[model_name]
                param_grid = self.param_grids[model_name]
                
                # Use stratified cross-validation
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring='roc_auc',
                    n_jobs=-1, verbose=0, error_score='raise'
                )
                
                grid_search.fit(X, y)
                
                tuning_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_,
                    'cv_results': {
                        'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                        'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
                    }
                }
                
                # Update model with best parameters
                self.models[model_name] = grid_search.best_estimator_
                self.trained_models[model_name] = grid_search.best_estimator_
                
                logger.info(f"✅ {model_name} tuned - Best ROC-AUC: {grid_search.best_score_:.4f}")
                logger.info(f"Best params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"❌ Error tuning {model_name}: {str(e)}")
                tuning_results[model_name] = {'error': str(e)}
        
        return tuning_results
    
    def save_models(self, model_dir: str = "models/"):
        """Save all trained models and preprocessing components"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.trained_models.items():
            model_path = f"{model_dir}{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"💾 Saved {model_name} to {model_path}")
        
        # Save preprocessing components
        preprocessing_components = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': True
        }
        
        preprocessing_path = f"{model_dir}preprocessing_components.joblib"
        joblib.dump(preprocessing_components, preprocessing_path)
        
        # Save model performances
        performance_path = f"{model_dir}model_performances.json"
        with open(performance_path, 'w') as f:
            json.dump(self.model_performances, f, indent=2)
        
        logger.info(f"💾 Saved preprocessing components to {preprocessing_path}")
        logger.info(f"💾 Saved performances to {performance_path}")

def main():
    """Enhanced main function with comprehensive error handling"""
    parser = argparse.ArgumentParser(description='Enhanced Loan Prediction Model Training')
    parser.add_argument('--models', nargs='+', 
                       choices=['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'logistic_regression'],
                       help='Models to train')
    parser.add_argument('--data-file', default='data/loan_training_dataset.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--output-dir', default='models/',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Enhanced Model Training Started")
    print("=" * 50)
    print(f"📊 Data file: {args.data_file}")
    print(f"🤖 Models: {args.models or 'all available'}")
    print(f"🎯 Hyperparameter tuning: {'Yes' if args.hyperparameter_tuning else 'No'}")
    print(f"📈 Test size: {args.test_size}")
    print(f"🎲 Random state: {args.random_state}")
    
    try:
        # Load and validate dataset
        if not os.path.exists(args.data_file):
            raise FileNotFoundError(f"Dataset not found: {args.data_file}")
        
        df = pd.read_csv(args.data_file)
        logger.info(f"📊 Loaded dataset: {len(df)} records, {len(df.columns)} columns")
        
        # Validate dataset
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if 'loan_status' not in df.columns:
            raise ValueError("Target column 'loan_status' not found")
        
        # Initialize trainer
        trainer = RobustModelTrainer(
            models_to_train=args.models,
            random_state=args.random_state
        )
        
        if not trainer.models:
            raise ValueError("No models available for training")
        
        print(f"🤖 Available models: {list(trainer.models.keys())}")
        
        # Prepare data
        X, y, feature_names = trainer.prepare_data(df)
        
        # Train models
        results = trainer.train_models(X, y, test_size=args.test_size)
        
        # Hyperparameter tuning if requested
        tuning_results = {}
        if args.hyperparameter_tuning:
            successful_models = [name for name in results.keys() 
                               if 'error' not in results[name] and name != 'training_summary']
            if successful_models:
                tuning_results = trainer.hyperparameter_tuning(X, y, successful_models)
                
                # Re-train with optimized parameters
                print("🔄 Re-training with optimized parameters...")
                final_results = trainer.train_models(X, y, test_size=args.test_size)
                results.update(final_results)
        
        # Save all models and components
        trainer.save_models(args.output_dir)
        
        # Generate comprehensive training report
        report = {
            'training_summary': results.get('training_summary', {}),
            'model_performances': trainer.model_performances,
            'hyperparameter_tuning': tuning_results,
            'training_config': {
                'data_file': args.data_file,
                'models_trained': list(trainer.trained_models.keys()),
                'test_size': args.test_size,
                'random_state': args.random_state,
                'hyperparameter_tuning': args.hyperparameter_tuning
            },
            'dataset_info': {
                'samples': len(df),
                'features': len(feature_names),
                'approval_rate': (df['loan_status'] == 'Y').mean(),
                'feature_names': feature_names
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save training report
        report_path = f"{args.output_dir}training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print results summary
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        if 'training_summary' in results:
            summary = results['training_summary']
            print(f"🏆 Best Model: {summary['best_model']}")
            print(f"📈 Best ROC-AUC Score: {summary['best_score']:.4f}")
            print(f"🤖 Models Trained: {summary['total_models']}")
            print(f"🔢 Features Used: {summary['feature_count']}")
        
        print(f"\n📊 Model Performance Comparison:")
        print("-" * 40)
        for model_name, performance in trainer.model_performances.items():
            print(f"{model_name:20} | ROC-AUC: {performance['roc_auc']:.4f} | "
                  f"Accuracy: {performance['accuracy']:.4f} | CV: {performance['cv_mean']:.4f}")
        
        print(f"\n📁 Files Created:")
        print(f"   📋 Training Report: {report_path}")
        print(f"   🤖 Model Files: {args.output_dir}*_model.joblib")
        print(f"   🔧 Preprocessing: {args.output_dir}preprocessing_components.joblib")
        print(f"   📊 Performances: {args.output_dir}model_performances.json")
        
        print(f"\n🚀 Next Steps:")
        print("   1. Start the API server: uvicorn app.main:app --reload")
        print("   2. Test predictions: python scripts/quick_test.py")
        print("   3. Switch models via API: POST /model/switch")
        print("   4. Monitor performance: GET /model/comparison")
        
        return trainer, results, report
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        print("💡 Generate data first: python scripts/generate_loan_data.py")
        return None, None, None
        
    except ValueError as e:
        logger.error(f"❌ Data validation error: {e}")
        print("💡 Check your dataset format and columns")
        return None, None, None
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print("💡 Check logs/training.log for detailed error information")
        return None, None, None

if __name__ == "__main__":
    main()