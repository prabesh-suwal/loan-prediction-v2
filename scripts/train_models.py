# scripts/train_models.py - Complete training script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
import json
import os
import argparse

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models_to_train=None):
        self.models = {}
        
        # Initialize available models
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        self.models.update({
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        })
        
        # Filter models if specified
        if models_to_train:
            self.models = {k: v for k, v in self.models.items() if k in models_to_train}
        
        self.param_grids = {
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.15],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.15],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        self.trained_models = {}
        self.model_performances = {}
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training with comprehensive preprocessing"""
        
        logger.info("🔄 Preparing data for training...")
        
        # Separate features and target
        target_col = 'loan_status'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove string columns that shouldn't be used for ML (name, email)
        columns_to_remove = ['name', 'email']
        for col in columns_to_remove:
            if col in X.columns:
                X = X.drop(columns=[col])
                logger.info(f"Removed string column: {col}")
        
        # Convert target to binary
        y = (y == 'Y').astype(int)
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'bool']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Feature engineering
        X = self._engineer_features(X)
        
        # Handle missing values ONLY for numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].median())
        
        # Replace any remaining NaN with 0 for categorical columns
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"✅ Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"📊 Target distribution: {np.bincount(y)} (0: Reject, 1: Approve)")
        
        return X_scaled, y, X.columns.tolist()
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        
        # Income ratios
        X['total_income'] = X['applicant_income'] + X['coapplicant_income']
        X['loan_to_income_ratio'] = X['loan_amount'] / X['total_income']
        X['monthly_income'] = X['total_income'] / 12
        X['monthly_emi_estimated'] = X['loan_amount'] / X['loan_amount_term']
        X['emi_to_income_ratio'] = X['monthly_emi_estimated'] / X['monthly_income']
        
        # Expense ratios
        X['expense_to_income_ratio'] = X['monthly_expenses'] / X['monthly_income']
        X['total_obligations'] = X['monthly_expenses'] + X['other_emis']
        X['obligation_to_income_ratio'] = X['total_obligations'] / X['monthly_income']
        
        # Credit utilization and risk factors
        X['credit_utilization_estimated'] = X['no_of_credit_cards'] * 50000 * 0.3  # Estimated usage
        X['credit_age_score'] = X['age'] * 10 + X['credit_score']
        X['employment_stability_score'] = X['years_in_current_job'] * 10
        
        # Asset ratios
        X['collateral_to_loan_ratio'] = np.where(
            X['loan_amount'] > 0,
            X['collateral_value'] / X['loan_amount'],
            0
        )
        
        X['savings_months'] = np.where(
            X['monthly_expenses'] > 0,
            X['bank_balance'] / X['monthly_expenses'],
            0
        )
        
        # Risk composite scores
        X['financial_stability_score'] = (
            (X['total_income'] / 100000) +
            (X['years_in_current_job'] * 2) +
            (X['credit_score'] / 100) +
            (5 - X['loan_default_history']) +
            (X['has_life_insurance'].astype(int) * 2)
        )
        
        X['risk_factor_count'] = (
            (X['loan_default_history'] > 0).astype(int) +
            (X['avg_payment_delay_days'] > 30).astype(int) +
            (X['credit_score'] < 650).astype(int) +
            (X['obligation_to_income_ratio'] > 0.6).astype(int) +
            (X['region_default_rate'] > 8).astype(int)
        )
        
        return X
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Train all specified models and return comprehensive results"""
        
        logger.info(f"🤖 Starting model training for: {list(self.models.keys())}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"🔄 Training {model_name}...")
            
            try:
                # Train model
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(
                        self.feature_names[:len(model.feature_importances_)],
                        model.feature_importances_
                    ))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(
                        self.feature_names[:len(model.coef_[0])],
                        np.abs(model.coef_[0])
                    ))
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'feature_importance': feature_importance,
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist()
                }
                
                # Store trained model
                self.trained_models[model_name] = model
                self.model_performances[model_name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean()
                }
                
                logger.info(f"✅ {model_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, CV: {cv_scores.mean():.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Identify best model
        successful_models = [name for name in results.keys() if 'error' not in results[name]]
        if successful_models:
            best_model_name = max(successful_models, key=lambda name: results[name]['roc_auc'])
            
            results['best_model'] = best_model_name
            results['training_summary'] = {
                'total_models': len(successful_models),
                'best_model': best_model_name,
                'best_score': results[best_model_name]['roc_auc'],
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            }
            
            logger.info(f"🏆 Best model: {best_model_name} with ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, models_to_tune: List[str] = None) -> Dict[str, Any]:
        """Perform hyperparameter tuning for specified models"""
        
        if models_to_tune is None:
            # Tune all available models
            models_to_tune = list(self.models.keys())
        
        logger.info(f"🎯 Starting hyperparameter tuning for: {models_to_tune}")
        
        tuning_results = {}
        
        for model_name in models_to_tune:
            if model_name not in self.models:
                continue
                
            logger.info(f"🔧 Tuning hyperparameters for {model_name}...")
            
            try:
                model = self.models[model_name]
                param_grid = self.param_grids.get(model_name, {})
                
                if not param_grid:
                    logger.warning(f"No parameter grid defined for {model_name}, skipping...")
                    continue
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, scoring='roc_auc', 
                    n_jobs=-1, verbose=1
                )
                
                grid_search.fit(X, y)
                
                tuning_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_
                }
                
                # Update model with best parameters
                self.models[model_name] = grid_search.best_estimator_
                self.trained_models[model_name] = grid_search.best_estimator_
                
                logger.info(f"✅ {model_name} tuned - Best score: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error tuning {model_name}: {str(e)}")
                tuning_results[model_name] = {'error': str(e)}
        
        return tuning_results
    
    def save_models(self, model_dir: str = "models/"):
        """Save all trained models and preprocessing components"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = f"{model_dir}{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"💾 Saved {model_name} model to {model_path}")
        
        # Save preprocessing components
        preprocessing_components = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        preprocessing_path = f"{model_dir}preprocessing_components.joblib"
        joblib.dump(preprocessing_components, preprocessing_path)
        
        # Save model performances
        performance_path = f"{model_dir}model_performances.json"
        with open(performance_path, 'w') as f:
            json.dump(self.model_performances, f, indent=2)
        
        logger.info(f"💾 Saved preprocessing components to {preprocessing_path}")
        logger.info(f"💾 Saved model performances to {performance_path}")

def main():
    """Main training function with command line arguments"""
    parser = argparse.ArgumentParser(description='Train loan prediction models')
    parser.add_argument('--models', nargs='+', 
                       choices=['xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression'],
                       help='Models to train')
    parser.add_argument('--data-file', default='data/loan_training_dataset.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--output-dir', default='models/',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    print("🚀 Starting Model Training...")
    print(f"📊 Data file: {args.data_file}")
    print(f"🤖 Models to train: {args.models or 'all available'}")
    print(f"🎯 Hyperparameter tuning: {'Yes' if args.hyperparameter_tuning else 'No'}")
    
    # Load dataset
    try:
        df = pd.read_csv(args.data_file)
        print(f"📊 Loaded dataset: {len(df)} records")
    except FileNotFoundError:
        print(f"❌ Dataset not found: {args.data_file}")
        print("💡 Generate data first with: python scripts/generate_loan_data.py")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(models_to_train=args.models)
    
    # Check if any models are available
    if not trainer.models:
        print("❌ No models available to train")
        if not XGBOOST_AVAILABLE:
            print("💡 Install XGBoost with: pip install xgboost")
        return
    
    print(f"🤖 Available models: {list(trainer.models.keys())}")
    
    # Prepare data
    X, y, feature_names = trainer.prepare_data(df)
    
    # Train models
    results = trainer.train_models(X, y)
    
    # Hyperparameter tuning if requested
    tuning_results = {}
    if args.hyperparameter_tuning:
        successful_models = [name for name in results.keys() if 'error' not in results[name]]
        if successful_models:
            tuning_results = trainer.hyperparameter_tuning(X, y, successful_models)
            
            # Re-train with tuned parameters
            print("🔄 Re-training with optimized parameters...")
            final_results = trainer.train_models(X, y)
            results.update(final_results)
    
    # Save models
    trainer.save_models(args.output_dir)
    
    # Generate and save comprehensive report
    report = {
        'training_summary': results.get('training_summary', {}),
        'model_performances': trainer.model_performances,
        'hyperparameter_tuning': tuning_results,
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'file': args.data_file,
            'samples': len(df),
            'features': len(feature_names),
            'approval_rate': (df['loan_status'] == 'Y').mean()
        }
    }
    
    # Save training report
    report_path = f"{args.output_dir}training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n🎉 Training completed successfully!")
    if 'training_summary' in results:
        print(f"🏆 Best model: {results['training_summary']['best_model']}")
        print(f"📈 Best ROC-AUC: {results['training_summary']['best_score']:.4f}")
    print(f"📋 Training report saved: {report_path}")
    
    # Print model comparison
    print(f"\n📊 Model Performance Summary:")
    for model_name, performance in trainer.model_performances.items():
        print(f"   {model_name:20} - ROC-AUC: {performance['roc_auc']:.4f}, Accuracy: {performance['accuracy']:.4f}")
    
    return trainer, results, report

if __name__ == "__main__":
    main()