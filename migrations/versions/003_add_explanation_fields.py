# migrations/versions/add_explanation_fields.py
"""Add explanation fields to loan_applications table

Revision ID: add_explanation_fields
Revises: previous_revision_id
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_add_explanation_fields'
down_revision = '002_initial_data'  # Replace with actual previous revision
branch_labels = None
depends_on = None

def upgrade():
    """Add explanation fields to loan_applications table"""
    
    # Add explanation text fields
    op.add_column('loan_applications', 
                  sa.Column('decision_summary', sa.Text(), nullable=True))
    op.add_column('loan_applications', 
                  sa.Column('detailed_explanation', sa.Text(), nullable=True))
    op.add_column('loan_applications', 
                  sa.Column('plain_text_summary', sa.Text(), nullable=True))
    
    # Add explanation JSON fields
    op.add_column('loan_applications', 
                  sa.Column('risk_explanation', sa.JSON(), nullable=True))
    op.add_column('loan_applications', 
                  sa.Column('key_factors', sa.JSON(), nullable=True))
    op.add_column('loan_applications', 
                  sa.Column('recommendations', sa.JSON(), nullable=True))
    op.add_column('loan_applications', 
                  sa.Column('next_steps', sa.JSON(), nullable=True))
    op.add_column('loan_applications', 
                  sa.Column('explanation_metadata', sa.JSON(), nullable=True))
    
    

def downgrade():
    """Remove explanation fields from loan_applications table"""
    
    # Remove explanation fields
    op.drop_column('loan_applications', 'explanation_metadata')
    op.drop_column('loan_applications', 'next_steps')
    op.drop_column('loan_applications', 'recommendations')
    op.drop_column('loan_applications', 'key_factors')
    op.drop_column('loan_applications', 'risk_explanation')
    op.drop_column('loan_applications', 'plain_text_summary')
    op.drop_column('loan_applications', 'detailed_explanation')
    op.drop_column('loan_applications', 'decision_summary')
    
    