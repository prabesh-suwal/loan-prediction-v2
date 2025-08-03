"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('role', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create field_weights table
    op.create_table('field_weights',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('field_name', sa.String(length=100), nullable=False),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('field_name')
    )
    op.create_index(op.f('ix_field_weights_id'), 'field_weights', ['id'], unique=False)

    # Create loan_applications table
    op.create_table('loan_applications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('gender', sa.String(length=10), nullable=False),
        sa.Column('married', sa.String(length=10), nullable=False),
        sa.Column('dependents', sa.Integer(), nullable=False),
        sa.Column('education', sa.String(length=20), nullable=False),
        sa.Column('age', sa.Integer(), nullable=False),
        sa.Column('children', sa.Integer(), nullable=True),
        sa.Column('spouse_employed', sa.Boolean(), nullable=True),
        sa.Column('self_employed', sa.String(length=10), nullable=False),
        sa.Column('employment_type', sa.String(length=20), nullable=True),
        sa.Column('years_in_current_job', sa.Float(), nullable=True),
        sa.Column('employer_category', sa.String(length=10), nullable=True),
        sa.Column('industry', sa.String(length=20), nullable=True),
        sa.Column('applicant_income', sa.Float(), nullable=False),
        sa.Column('coapplicant_income', sa.Float(), nullable=True),
        sa.Column('monthly_expenses', sa.Float(), nullable=False),
        sa.Column('other_emis', sa.Float(), nullable=True),
        sa.Column('loan_amount', sa.Float(), nullable=False),
        sa.Column('loan_amount_term', sa.Float(), nullable=False),
        sa.Column('loan_purpose', sa.String(length=20), nullable=True),
        sa.Column('requested_interest_rate', sa.Float(), nullable=True),
        sa.Column('credit_score', sa.Integer(), nullable=True),
        sa.Column('credit_history', sa.Integer(), nullable=False),
        sa.Column('no_of_credit_cards', sa.Integer(), nullable=True),
        sa.Column('loan_default_history', sa.Integer(), nullable=True),
        sa.Column('avg_payment_delay_days', sa.Float(), nullable=True),
        sa.Column('has_vehicle', sa.Boolean(), nullable=True),
        sa.Column('has_life_insurance', sa.Boolean(), nullable=True),
        sa.Column('property_area', sa.String(length=20), nullable=False),
        sa.Column('bank_account_type', sa.String(length=20), nullable=True),
        sa.Column('bank_balance', sa.Float(), nullable=True),
        sa.Column('savings_score', sa.Float(), nullable=True),
        sa.Column('collateral_type', sa.String(length=20), nullable=True),
        sa.Column('collateral_value', sa.Float(), nullable=True),
        sa.Column('city_tier', sa.String(length=20), nullable=True),
        sa.Column('pincode', sa.String(length=6), nullable=True),
        sa.Column('region_default_rate', sa.Float(), nullable=True),
        sa.Column('is_approved', sa.Boolean(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=True),
        sa.Column('prediction_details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('recommended_interest_rate', sa.Float(), nullable=True),
        sa.Column('conditions', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('model_used', sa.String(length=20), nullable=True),
        sa.Column('model_performance', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_loan_applications_id'), 'loan_applications', ['id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_loan_applications_id'), table_name='loan_applications')
    op.drop_table('loan_applications')
    op.drop_index(op.f('ix_field_weights_id'), table_name='field_weights')
    op.drop_table('field_weights')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')