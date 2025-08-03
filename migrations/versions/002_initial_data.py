"""Create initial users and field weights

Revision ID: 002_initial_data
Revises: 001_initial_schema
Create Date: 2024-01-01 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from sqlalchemy import String, Integer, Float, Boolean, DateTime
from datetime import datetime
import bcrypt

# revision identifiers, used by Alembic.
revision = '002_initial_data'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def upgrade():
    """Insert initial data"""
    
    # Define table structures for data insertion
    users_table = table('users',
        column('username', String),
        column('email', String),
        column('hashed_password', String),
        column('full_name', String),
        column('role', String),
        column('is_active', Boolean),
        column('created_at', DateTime),
        column('updated_at', DateTime)
    )
    
    field_weights_table = table('field_weights',
        column('field_name', String),
        column('weight', Float),
        column('is_active', Boolean),
        column('description', String),
        column('category', String),
        column('created_at', DateTime),
        column('updated_at', DateTime)
    )
    
    # Insert initial users
    current_time = datetime.utcnow()
    
    # Create superadmin user
    op.bulk_insert(users_table, [
        {
            'username': 'admin',
            'email': 'admin@loanpredict.com',
            'hashed_password': hash_password('Admin@123'),
            'full_name': 'System Administrator',
            'role': 'superadmin',
            'is_active': True,
            'created_at': current_time,
            'updated_at': current_time
        },
        {
            'username': 'rm_user',
            'email': 'rm@loanpredict.com',
            'hashed_password': hash_password('RM@123'),
            'full_name': 'Relationship Manager',
            'role': 'RM',
            'is_active': True,
            'created_at': current_time,
            'updated_at': current_time
        }
    ])
    
    # Insert default field weights
    default_weights = [
        # High importance factors
        {'field_name': 'credit_score', 'weight': 2.0, 'category': 'Credit', 'description': 'Credit bureau score - primary risk indicator'},
        {'field_name': 'credit_history', 'weight': 2.0, 'category': 'Credit', 'description': 'Credit history track record'},
        {'field_name': 'loan_default_history', 'weight': 1.8, 'category': 'Credit', 'description': 'Previous loan defaults'},
        {'field_name': 'applicant_income', 'weight': 1.5, 'category': 'Income', 'description': 'Primary applicant income'},
        {'field_name': 'loan_amount', 'weight': 1.5, 'category': 'Loan', 'description': 'Requested loan amount'},
        
        # Medium-high importance factors
        {'field_name': 'employment_type', 'weight': 1.3, 'category': 'Employment', 'description': 'Type of employment'},
        {'field_name': 'years_in_current_job', 'weight': 1.2, 'category': 'Employment', 'description': 'Employment stability'},
        {'field_name': 'collateral_value', 'weight': 1.3, 'category': 'Assets', 'description': 'Value of collateral provided'},
        {'field_name': 'coapplicant_income', 'weight': 1.2, 'category': 'Income', 'description': 'Co-applicant income'},
        {'field_name': 'monthly_expenses', 'weight': 1.1, 'category': 'Financial', 'description': 'Monthly financial obligations'},
        
        # Medium importance factors
        {'field_name': 'property_area', 'weight': 1.1, 'category': 'Location', 'description': 'Property location type'},
        {'field_name': 'education', 'weight': 1.1, 'category': 'Demographics', 'description': 'Educational qualification'},
        {'field_name': 'employer_category', 'weight': 1.1, 'category': 'Employment', 'description': 'Employer rating category'},
        {'field_name': 'industry', 'weight': 1.0, 'category': 'Employment', 'description': 'Industry sector'},
        {'field_name': 'loan_amount_term', 'weight': 1.0, 'category': 'Loan', 'description': 'Loan tenure in months'},
        
        # Lower-medium importance factors
        {'field_name': 'age', 'weight': 1.0, 'category': 'Demographics', 'description': 'Applicant age'},
        {'field_name': 'bank_balance', 'weight': 1.0, 'category': 'Financial', 'description': 'Current bank balance'},
        {'field_name': 'savings_score', 'weight': 0.9, 'category': 'Financial', 'description': 'Savings pattern score'},
        {'field_name': 'no_of_credit_cards', 'weight': 0.9, 'category': 'Credit', 'description': 'Number of credit cards'},
        {'field_name': 'avg_payment_delay_days', 'weight': 0.9, 'category': 'Credit', 'description': 'Average payment delays'},
        
        # Lower importance factors
        {'field_name': 'married', 'weight': 0.9, 'category': 'Demographics', 'description': 'Marital status'},
        {'field_name': 'dependents', 'weight': 0.9, 'category': 'Demographics', 'description': 'Number of dependents'},
        {'field_name': 'has_life_insurance', 'weight': 0.9, 'category': 'Assets', 'description': 'Life insurance coverage'},
        {'field_name': 'has_vehicle', 'weight': 0.8, 'category': 'Assets', 'description': 'Vehicle ownership'},
        {'field_name': 'gender', 'weight': 0.8, 'category': 'Demographics', 'description': 'Gender'},
        
        # Additional factors
        {'field_name': 'self_employed', 'weight': 1.0, 'category': 'Employment', 'description': 'Self-employment status'},
        {'field_name': 'spouse_employed', 'weight': 0.8, 'category': 'Income', 'description': 'Spouse employment status'},
        {'field_name': 'children', 'weight': 0.8, 'category': 'Demographics', 'description': 'Number of children'},
        {'field_name': 'other_emis', 'weight': 1.1, 'category': 'Financial', 'description': 'Existing EMI obligations'},
        {'field_name': 'loan_purpose', 'weight': 0.9, 'category': 'Loan', 'description': 'Purpose of loan'},
        {'field_name': 'bank_account_type', 'weight': 0.8, 'category': 'Financial', 'description': 'Type of bank account'},
        {'field_name': 'collateral_type', 'weight': 1.0, 'category': 'Assets', 'description': 'Type of collateral'},
        {'field_name': 'city_tier', 'weight': 0.9, 'category': 'Location', 'description': 'City tier classification'},
        {'field_name': 'region_default_rate', 'weight': 1.0, 'category': 'Location', 'description': 'Regional default rate'},
        {'field_name': 'requested_interest_rate', 'weight': 0.8, 'category': 'Loan', 'description': 'Requested interest rate'}
    ]
    
    # Prepare weight data for insertion
    weight_data = []
    for weight_config in default_weights:
        weight_data.append({
            'field_name': weight_config['field_name'],
            'weight': weight_config['weight'],
            'is_active': True,
            'description': weight_config['description'],
            'category': weight_config['category'],
            'created_at': current_time,
            'updated_at': current_time
        })
    
    # Insert field weights
    op.bulk_insert(field_weights_table, weight_data)


def downgrade():
    """Remove initial data"""
    
    # Remove initial users
    op.execute("DELETE FROM users WHERE username IN ('admin', 'rm_user')")
    
    # Remove initial field weights
    op.execute("DELETE FROM field_weights")