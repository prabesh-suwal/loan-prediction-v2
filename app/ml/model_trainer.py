from ..core.dependencies import get_current_user


# app/ml/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, Tuple
import logging
from .data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.best_score = 0
        self.preprocessor = DataPreprocessor()
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'loan_status') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target to binary if needed
        if y.dtype == 'object':
            y = (y == 'Y').astype(int)  # Assuming 'Y' means approved
        
        # Preprocess features
        X_processed = self.preprocessor.preprocess_loan_data(X.to_dict('records')[0])
        
        return X_processed, y.values
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train multiple models and return performance metrics"""
        
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = model.score(X_test, y_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Update best model
                score = roc_auc if roc_auc is not None else accuracy
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f if roc_auc else 'N/A'}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning for the best models"""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        tuned_results = {}
        
        for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            try:
                model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
                )
                
                grid_search.fit(X, y)
                
                tuned_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_
                }
                
                # Update model with best parameters
                self.models[model_name] = grid_search.best_estimator_
                
                logger.info(f"{model_name} best score: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
                tuned_results[model_name] = {'error': str(e)}
        
        return tuned_results
    
    def save_best_model(self, model_path: str, preprocessor_path: str):
        """Save the best trained model and preprocessor"""
        
        if self.best_model is None:
            raise ValueError("No trained model available to save")
        
        # Save model
        joblib.dump(self.best_model, model_path)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        logger.info(f"Best model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def generate_training_report(self, results: Dict[str, Any], tuned_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        report = {
            'training_summary': {
                'total_models_trained': len([r for r in results.values() if 'error' not in r]),
                'best_model': None,
                'best_score': self.best_score
            },
            'model_performance': {},
            'recommendations': []
        }
        
        # Find best performing model
        best_model_name = None
        best_score = 0
        
        for model_name, result in results.items():
            if 'error' not in result:
                score = result.get('roc_auc', result.get('accuracy', 0))
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        report['training_summary']['best_model'] = best_model_name
        
        # Add performance details
        for model_name, result in results.items():
            if 'error' not in result:
                report['model_performance'][model_name] = {
                    'accuracy': result['accuracy'],
                    'roc_auc': result['roc_auc'],
                    'cross_validation_mean': result['cv_mean'],
                    'cross_validation_std': result['cv_std']
                }
        
        # Add tuning results if available
        if tuned_results:
            report['hyperparameter_tuning'] = tuned_results
        
        # Generate recommendations
        if best_score > 0.8:
            report['recommendations'].append("Model performance is excellent (>80%)")
        elif best_score > 0.7:
            report['recommendations'].append("Model performance is good (>70%)")
        else:
            report['recommendations'].append("Consider collecting more data or feature engineering")
        
        return report
