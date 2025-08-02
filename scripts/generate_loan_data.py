import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

class LoanDataGenerator:
    def __init__(self, num_records=2500):
        self.num_records = num_records
        self.data = []
        
        # Define realistic distributions and correlations
        self.income_brackets = {
            'low': (20000, 40000),
            'middle': (40000, 80000),
            'upper_middle': (80000, 150000),
            'high': (150000, 300000)
        }
        
        self.city_tiers = {
            'Tier-1': {'cities': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'], 'weight': 0.3},
            'Tier-2': {'cities': ['Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur'], 'weight': 0.5},
            'Tier-3': {'cities': ['Nashik', 'Rajkot', 'Vadodara', 'Aurangabad', 'Dhanbad'], 'weight': 0.2}
        }
        
        self.industries = {
            'IT': {'stability': 0.85, 'income_multiplier': 1.3, 'weight': 0.25},
            'Finance': {'stability': 0.80, 'income_multiplier': 1.4, 'weight': 0.15},
            'Healthcare': {'stability': 0.90, 'income_multiplier': 1.2, 'weight': 0.10},
            'Government': {'stability': 0.95, 'income_multiplier': 1.0, 'weight': 0.12},
            'Manufacturing': {'stability': 0.70, 'income_multiplier': 1.1, 'weight': 0.15},
            'Retail': {'stability': 0.60, 'income_multiplier': 0.9, 'weight': 0.10},
            'Education': {'stability': 0.85, 'income_multiplier': 0.95, 'weight': 0.08},
            'Others': {'stability': 0.65, 'income_multiplier': 1.0, 'weight': 0.05}
        }
    
    def generate_realistic_dataset(self):
        """Generate comprehensive realistic loan dataset"""
        
        print(f"🔄 Generating {self.num_records} realistic loan applications...")
        
        for i in range(self.num_records):
            record = self._generate_single_record()
            self.data.append(record)
            
            if (i + 1) % 500 == 0:
                print(f"✅ Generated {i + 1} records...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        
        # Apply business logic for loan approval
        df = self._apply_approval_logic(df)
        
        print(f"🎉 Successfully generated {len(df)} loan applications!")
        print(f"📊 Approval rate: {(df['loan_status'] == 'Y').mean() * 100:.1f}%")
        
        return df
    
    def _generate_single_record(self):
        """Generate a single realistic loan record"""
        
        # Basic demographics
        age = np.random.randint(22, 65)
        gender = np.random.choice(['Male', 'Female'], p=[0.65, 0.35])
        
        # Marital status (correlated with age)
        if age < 25:
            married = np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
        elif age < 35:
            married = np.random.choice(['Yes', 'No'], p=[0.7, 0.3])
        else:
            married = np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
        
        # Education (correlated with income potential)
        education = np.random.choice(['Graduate', 'Not Graduate'], p=[0.7, 0.3])
        
        # Employment details
        industry = np.random.choice(
            list(self.industries.keys()),
            p=[self.industries[ind]['weight'] for ind in self.industries.keys()]
        )
        
        employment_type = self._get_employment_type(industry)
        self_employed = 'Yes' if employment_type in ['Self-Employed', 'Business Owner'] else 'No'
        
        # Years in current job (realistic distribution)
        if age < 25:
            years_in_job = np.random.exponential(1.5)
        elif age < 35:
            years_in_job = np.random.exponential(3.0)
        else:
            years_in_job = np.random.exponential(5.0)
        years_in_job = min(years_in_job, age - 20)  # Can't work more than age allows
        
        # Employer category (correlated with industry and employment type)
        employer_category = self._get_employer_category(industry, employment_type)
        
        # Income generation (correlated with education, industry, age, location)
        base_income = self._generate_income(age, education, industry, employment_type)
        applicant_income = base_income
        
        # Co-applicant income (if married)
        if married == 'Yes':
            coapplicant_income = np.random.exponential(40000) if np.random.random() < 0.6 else 0
            spouse_employed = coapplicant_income > 0
        else:
            coapplicant_income = 0
            spouse_employed = False
        
        # Dependents and children
        if married == 'Yes':
            dependents = np.random.poisson(1.2)
            children = min(dependents, np.random.poisson(1.5))
        else:
            dependents = np.random.poisson(0.3)
            children = 0
        
        dependents = min(dependents, 5)  # Cap at 5
        children = min(children, dependents)
        
        # Location details
        city_tier = np.random.choice(
            list(self.city_tiers.keys()),
            p=[self.city_tiers[tier]['weight'] for tier in self.city_tiers.keys()]
        )
        
        property_area = self._get_property_area(city_tier)
        
        # Generate pincode based on city tier
        if city_tier == 'Tier-1':
            pincode = f"{np.random.randint(100000, 700000):06d}"
        elif city_tier == 'Tier-2':
            pincode = f"{np.random.randint(200000, 800000):06d}"
        else:
            pincode = f"{np.random.randint(100000, 900000):06d}"
        
        # Regional default rate (varies by tier and random factors)
        base_default_rate = {'Tier-1': 3.5, 'Tier-2': 5.5, 'Tier-3': 8.0}
        region_default_rate = base_default_rate[city_tier] + np.random.normal(0, 1.5)
        region_default_rate = max(1.0, min(region_default_rate, 15.0))
        
        # Monthly expenses (realistic based on income and location)
        total_income = applicant_income + coapplicant_income
        expense_ratio = self._get_expense_ratio(city_tier, dependents)
        monthly_expenses = total_income * expense_ratio / 12
        
        # Credit history and behavior
        credit_score = self._generate_credit_score(age, years_in_job, industry, total_income)
        credit_history = 1 if credit_score > 600 else 0
        
        # Credit cards and defaults (correlated with credit score and income)
        no_of_credit_cards = self._generate_credit_cards(credit_score, total_income)
        loan_default_history = self._generate_default_history(credit_score)
        avg_payment_delay_days = self._generate_payment_delays(credit_score)
        
        # Assets and lifestyle
        has_vehicle = self._determine_vehicle_ownership(total_income, city_tier)
        has_life_insurance = self._determine_insurance(age, married, total_income)
        
        # Banking details
        bank_account_type = self._get_bank_account_type(total_income)
        bank_balance = self._generate_bank_balance(total_income, monthly_expenses)
        savings_score = self._calculate_savings_score(bank_balance, total_income)
        
        # Loan details
        loan_amount, loan_term, loan_purpose = self._generate_loan_details(
            total_income, age, property_area, has_vehicle
        )
        
        # Interest rate request (based on credit score and loan amount)
        requested_interest_rate = self._generate_requested_rate(credit_score, loan_amount, total_income)
        
        # Existing EMIs
        other_emis = self._generate_existing_emis(total_income, age, credit_score)
        
        # Collateral
        collateral_type, collateral_value = self._generate_collateral(loan_amount, loan_purpose, total_income)
        
        # Generate realistic name and email
        name = fake.name()
        email = f"{name.lower().replace(' ', '.')}@{fake.domain_name()}"
        
        return {
            # Personal Information
            'name': name,
            'email': email,
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'age': age,
            'children': children,
            'spouse_employed': spouse_employed,
            
            # Employment & Stability
            'self_employed': self_employed,
            'employment_type': employment_type,
            'years_in_current_job': round(years_in_job, 1),
            'employer_category': employer_category,
            'industry': industry,
            
            # Income & Expenses
            'applicant_income': round(applicant_income),
            'coapplicant_income': round(coapplicant_income),
            'monthly_expenses': round(monthly_expenses),
            'other_emis': round(other_emis),
            
            # Loan Details
            'loan_amount': round(loan_amount),
            'loan_amount_term': loan_term,
            'loan_purpose': loan_purpose,
            'requested_interest_rate': requested_interest_rate,
            
            # Credit History & Behavior
            'credit_score': int(credit_score),
            'credit_history': credit_history,
            'no_of_credit_cards': no_of_credit_cards,
            'loan_default_history': loan_default_history,
            'avg_payment_delay_days': round(avg_payment_delay_days, 1),
            
            # Assets & Lifestyle
            'has_vehicle': has_vehicle,
            'has_life_insurance': has_life_insurance,
            'property_area': property_area,
            
            # Banking Info
            'bank_account_type': bank_account_type,
            'bank_balance': round(bank_balance),
            'savings_score': round(savings_score, 1),
            
            # Collateral
            'collateral_type': collateral_type,
            'collateral_value': round(collateral_value),
            
            # Geographic Info
            'city_tier': city_tier,
            'pincode': pincode,
            'region_default_rate': round(region_default_rate, 1)
        }
    
    def _get_employment_type(self, industry):
        """Get employment type based on industry"""
        if industry == 'Government':
            return 'Government'
        elif industry in ['IT', 'Finance', 'Healthcare']:
            return np.random.choice(['Salaried', 'Self-Employed'], p=[0.8, 0.2])
        elif industry == 'Manufacturing':
            return np.random.choice(['Salaried', 'Government'], p=[0.7, 0.3])
        else:
            return np.random.choice(['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer'], 
                                  p=[0.4, 0.3, 0.2, 0.1])
    
    def _get_employer_category(self, industry, employment_type):
        """Get employer category based on industry and employment type"""
        if employment_type == 'Government':
            return 'A'
        elif industry in ['IT', 'Finance']:
            return np.random.choice(['A', 'B', 'MNC'], p=[0.3, 0.4, 0.3])
        elif industry == 'Healthcare':
            return np.random.choice(['A', 'B'], p=[0.6, 0.4])
        else:
            return np.random.choice(['B', 'C', 'SME'], p=[0.4, 0.4, 0.2])
    
    def _generate_income(self, age, education, industry, employment_type):
        """Generate realistic income based on demographics"""
        base_income = 30000
        
        # Age factor
        if age < 25:
            age_multiplier = 0.7
        elif age < 35:
            age_multiplier = 1.0 + (age - 25) * 0.05
        elif age < 50:
            age_multiplier = 1.5 + (age - 35) * 0.02
        else:
            age_multiplier = 1.8
        
        # Education factor
        education_multiplier = 1.3 if education == 'Graduate' else 1.0
        
        # Industry factor
        industry_multiplier = self.industries[industry]['income_multiplier']
        
        # Employment type factor
        if employment_type == 'Government':
            employment_multiplier = 1.1
        elif employment_type == 'Self-Employed':
            employment_multiplier = np.random.uniform(0.8, 1.8)  # High variance
        elif employment_type == 'Business Owner':
            employment_multiplier = np.random.uniform(1.2, 3.0)  # Even higher variance
        else:
            employment_multiplier = 1.0
        
        income = base_income * age_multiplier * education_multiplier * industry_multiplier * employment_multiplier
        
        # Add some randomness
        income *= np.random.uniform(0.8, 1.2)
        
        return max(income, 15000)  # Minimum income
    
    def _get_property_area(self, city_tier):
        """Get property area based on city tier"""
        if city_tier == 'Tier-1':
            return np.random.choice(['Urban', 'Semiurban'], p=[0.8, 0.2])
        elif city_tier == 'Tier-2':
            return np.random.choice(['Urban', 'Semiurban', 'Rural'], p=[0.5, 0.4, 0.1])
        else:
            return np.random.choice(['Semiurban', 'Rural'], p=[0.6, 0.4])
    
    def _get_expense_ratio(self, city_tier, dependents):
        """Calculate expense ratio based on location and family size"""
        base_ratio = {'Tier-1': 0.7, 'Tier-2': 0.6, 'Tier-3': 0.5}
        dependent_factor = 1 + (dependents * 0.1)
        return min(base_ratio[city_tier] * dependent_factor, 0.85)
    
    def _generate_credit_score(self, age, years_in_job, industry, income):
        """Generate realistic credit score"""
        base_score = 650
        
        # Age factor (older people tend to have better credit)
        age_boost = (age - 25) * 2 if age > 25 else 0
        
        # Job stability factor
        stability_boost = min(years_in_job * 5, 50)
        
        # Industry factor
        industry_boost = self.industries[industry]['stability'] * 50 - 25
        
        # Income factor
        income_boost = min((income - 30000) / 1000, 100)
        
        score = base_score + age_boost + stability_boost + industry_boost + income_boost
        
        # Add randomness
        score += np.random.normal(0, 40)
        
        return max(300, min(score, 850))
    
    def _generate_credit_cards(self, credit_score, income):
        """Generate number of credit cards based on credit score and income"""
        if credit_score < 600:
            return np.random.poisson(0.5)
        elif credit_score < 700:
            return np.random.poisson(1.5)
        elif credit_score < 750:
            return np.random.poisson(2.5)
        else:
            return np.random.poisson(3.0) + 1
    
    def _generate_default_history(self, credit_score):
        """Generate loan default history based on credit score"""
        if credit_score > 750:
            return 0 if np.random.random() < 0.95 else 1
        elif credit_score > 700:
            return np.random.poisson(0.1)
        elif credit_score > 650:
            return np.random.poisson(0.3)
        elif credit_score > 600:
            return np.random.poisson(0.8)
        else:
            return np.random.poisson(1.5)
    
    def _generate_payment_delays(self, credit_score):
        """Generate average payment delay days"""
        if credit_score > 750:
            return np.random.exponential(2)
        elif credit_score > 700:
            return np.random.exponential(5)
        elif credit_score > 650:
            return np.random.exponential(10)
        elif credit_score > 600:
            return np.random.exponential(20)
        else:
            return np.random.exponential(30)
    
    def _determine_vehicle_ownership(self, income, city_tier):
        """Determine vehicle ownership based on income and location"""
        base_prob = {'Tier-1': 0.6, 'Tier-2': 0.7, 'Tier-3': 0.8}
        income_factor = min((income - 30000) / 100000, 0.4)
        prob = base_prob[city_tier] + income_factor
        return np.random.random() < prob
    
    def _determine_insurance(self, age, married, income):
        """Determine life insurance ownership"""
        base_prob = 0.3
        if age > 30:
            base_prob += 0.2
        if married == 'Yes':
            base_prob += 0.3
        if income > 60000:
            base_prob += 0.2
        return np.random.random() < min(base_prob, 0.9)
    
    def _get_bank_account_type(self, income):
        """Get bank account type based on income"""
        if income < 30000:
            return np.random.choice(['Basic', 'Savings'], p=[0.7, 0.3])
        elif income < 80000:
            return np.random.choice(['Savings', 'Premium'], p=[0.8, 0.2])
        elif income < 150000:
            return np.random.choice(['Savings', 'Premium', 'Current'], p=[0.5, 0.4, 0.1])
        else:
            return np.random.choice(['Premium', 'Current'], p=[0.6, 0.4])
    
    def _generate_bank_balance(self, income, monthly_expenses):
        """Generate realistic bank balance"""
        monthly_income = income / 12
        surplus = monthly_income - monthly_expenses
        
        if surplus > 0:
            # Can save money
            months_saved = np.random.exponential(6)
            balance = surplus * months_saved
        else:
            # Living paycheck to paycheck
            balance = np.random.exponential(monthly_income * 0.1)
        
        # Add some randomness
        balance *= np.random.uniform(0.5, 2.0)
        
        return max(balance, 1000)  # Minimum balance
    
    def _calculate_savings_score(self, bank_balance, income):
        """Calculate savings score as percentage of monthly income"""
        monthly_income = income / 12
        return min((bank_balance / monthly_income) * 10, 100)
    
    def _generate_loan_details(self, income, age, property_area, has_vehicle):
        """Generate realistic loan amount, term, and purpose"""
        
        # Loan purpose based on demographics
        if age < 30:
            purposes = ['Personal', 'Education', 'Vehicle']
            weights = [0.5, 0.3, 0.2]
        elif age < 40:
            purposes = ['Home', 'Personal', 'Vehicle', 'Business']
            weights = [0.4, 0.3, 0.2, 0.1]
        else:
            purposes = ['Home', 'Personal', 'Business', 'Medical']
            weights = [0.5, 0.25, 0.15, 0.1]
        
        loan_purpose = np.random.choice(purposes, p=weights)
        
        # Loan amount based on purpose and income
        annual_income = income
        
        if loan_purpose == 'Home':
            loan_amount = annual_income * np.random.uniform(3, 8)
            loan_term = np.random.choice([240, 300, 360], p=[0.2, 0.3, 0.5])
        elif loan_purpose == 'Vehicle':
            loan_amount = annual_income * np.random.uniform(0.5, 2)
            loan_term = np.random.choice([36, 48, 60, 72], p=[0.2, 0.3, 0.3, 0.2])
        elif loan_purpose == 'Education':
            loan_amount = annual_income * np.random.uniform(0.5, 3)
            loan_term = np.random.choice([60, 84, 120], p=[0.3, 0.4, 0.3])
        elif loan_purpose == 'Business':
            loan_amount = annual_income * np.random.uniform(1, 5)
            loan_term = np.random.choice([60, 120, 180, 240], p=[0.3, 0.3, 0.2, 0.2])
        else:  # Personal
            loan_amount = annual_income * np.random.uniform(0.2, 2)
            loan_term = np.random.choice([12, 24, 36, 48, 60], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        
        return loan_amount, loan_term, loan_purpose
    
    def _generate_requested_rate(self, credit_score, loan_amount, income):
        """Generate requested interest rate based on profile"""
        base_rate = 12.0
        
        # Credit score adjustment
        if credit_score > 750:
            rate_adjustment = -2.0
        elif credit_score > 700:
            rate_adjustment = -1.0
        elif credit_score > 650:
            rate_adjustment = 0.0
        elif credit_score > 600:
            rate_adjustment = 1.0
        else:
            rate_adjustment = 2.5
        
        # Loan size adjustment (smaller loans often have higher rates)
        if loan_amount < 100000:
            rate_adjustment += 1.0
        elif loan_amount > 1000000:
            rate_adjustment -= 0.5
        
        rate = base_rate + rate_adjustment + np.random.uniform(-1, 1)
        return round(max(6.0, min(rate, 24.0)), 1)
    
    def _generate_existing_emis(self, income, age, credit_score):
        """Generate existing EMI obligations"""
        monthly_income = income / 12
        
        # Probability of having existing EMIs
        if age < 25:
            emi_prob = 0.2
        elif age < 35:
            emi_prob = 0.5
        else:
            emi_prob = 0.7
        
        if credit_score < 600:
            emi_prob *= 0.5  # Bad credit = fewer loans
        
        if np.random.random() < emi_prob:
            # Has existing EMIs
            max_emi = monthly_income * 0.4  # 40% of income max
            existing_emi = np.random.uniform(0.1, 0.8) * max_emi
            return existing_emi
        else:
            return 0
    
    def _generate_collateral(self, loan_amount, loan_purpose, income):
        """Generate collateral details based on loan"""
        
        if loan_purpose == 'Home':
            # Home loans often have property as collateral
            collateral_type = 'Property'
            collateral_value = loan_amount * np.random.uniform(1.2, 2.0)
        elif loan_purpose == 'Vehicle':
            # Vehicle loans have vehicle as collateral
            collateral_type = 'Vehicle'
            collateral_value = loan_amount * np.random.uniform(1.0, 1.3)
        elif loan_amount > income * 2:  # Large loans need collateral
            collateral_options = ['Property', 'Fixed Deposit']
            collateral_type = np.random.choice(collateral_options)
            if collateral_type == 'Property':
                collateral_value = loan_amount * np.random.uniform(1.5, 3.0)
            else:  # Fixed Deposit
                collateral_value = loan_amount * np.random.uniform(1.1, 1.5)
        else:
            # No collateral for smaller loans
            collateral_type = 'None'
            collateral_value = 0
        
        return collateral_type, collateral_value
    
    def _apply_approval_logic(self, df):
        """Apply realistic loan approval logic"""
        
        def calculate_approval_probability(row):
            score = 0.5  # Base probability
            
            # Credit score factor (most important)
            if row['credit_score'] > 750:
                score += 0.35
            elif row['credit_score'] > 700:
                score += 0.25
            elif row['credit_score'] > 650:
                score += 0.10
            elif row['credit_score'] > 600:
                score -= 0.05
            else:
                score -= 0.25
            
            # Income stability
            total_income = row['applicant_income'] + row['coapplicant_income']
            loan_to_income = row['loan_amount'] / total_income
            
            if loan_to_income < 3:
                score += 0.2
            elif loan_to_income < 5:
                score += 0.1
            elif loan_to_income < 8:
                score -= 0.1
            else:
                score -= 0.3
            
            # Employment stability
            if row['employment_type'] == 'Government':
                score += 0.15
            elif row['years_in_current_job'] > 5:
                score += 0.1
            elif row['years_in_current_job'] < 1:
                score -= 0.1
            
            # Existing obligations
            monthly_income = total_income / 12
            if monthly_income > 0:
                obligation_ratio = (row['monthly_expenses'] + row['other_emis']) / monthly_income
                if obligation_ratio < 0.5:
                    score += 0.1
                elif obligation_ratio > 0.8:
                    score -= 0.2
            
            # Default history
            if row['loan_default_history'] == 0:
                score += 0.1
            else:
                score -= row['loan_default_history'] * 0.15
            
            # Collateral
            if row['collateral_value'] > row['loan_amount']:
                score += 0.15
            elif row['collateral_value'] > 0:
                score += 0.05
            
            # Education
            if row['education'] == 'Graduate':
                score += 0.05
            
            # Regional factors
            if row['region_default_rate'] < 5:
                score += 0.05
            elif row['region_default_rate'] > 10:
                score -= 0.1
            
            return max(0.05, min(score, 0.95))  # Keep between 5% and 95%
        
        # Calculate approval probabilities
        df['approval_probability'] = df.apply(calculate_approval_probability, axis=1)
        
        # Make approval decisions
        df['loan_status'] = df['approval_probability'].apply(
            lambda prob: 'Y' if np.random.random() < prob else 'N'
        )
        
        # Remove the probability column (not needed in final dataset)
        df = df.drop('approval_probability', axis=1)
        
        return df
    
    def save_dataset(self, df, filename='loan_dataset.csv'):
        """Save the generated dataset"""
        df.to_csv(filename, index=False)
        print(f"💾 Dataset saved as: {filename}")
        
        # Save summary statistics
        summary = {
            'total_records': len(df),
            'approval_rate': (df['loan_status'] == 'Y').mean(),
            'avg_loan_amount': df['loan_amount'].mean(),
            'avg_applicant_income': df['applicant_income'].mean(),
            'avg_credit_score': df['credit_score'].mean(),
            'education_distribution': df['education'].value_counts().to_dict(),
            'employment_type_distribution': df['employment_type'].value_counts().to_dict(),
            'loan_purpose_distribution': df['loan_purpose'].value_counts().to_dict(),
            'city_tier_distribution': df['city_tier'].value_counts().to_dict()
        }
        
        with open(filename.replace('.csv', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("📊 Dataset summary:")
        print(f"   Total records: {summary['total_records']}")
        print(f"   Approval rate: {summary['approval_rate']:.1%}")
        print(f"   Avg loan amount: ₹{summary['avg_loan_amount']:,.0f}")
        print(f"   Avg income: ₹{summary['avg_applicant_income']:,.0f}")
        print(f"   Avg credit score: {summary['avg_credit_score']:.0f}")
        
        return summary

def main():
    """Main function to generate and save loan dataset"""
    print("🚀 Starting Loan Dataset Generation...")
    
    # Generate dataset
    generator = LoanDataGenerator(num_records=2500)
    df = generator.generate_realistic_dataset()
    
    # Save dataset
    generator.save_dataset(df, 'data/loan_training_dataset.csv')
    
    # Display sample records
    print("\n📋 Sample records:")
    print(df.head(3).to_string())
    
    # Data quality checks
    print("\n🔍 Data Quality Checks:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate records: {df.duplicated().sum()}")
    print(f"   Data types: {len(df.dtypes.unique())} unique types")
    
    # Feature correlation analysis
    print("\n📈 Key Correlations with Loan Approval:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    loan_approved = (df['loan_status'] == 'Y').astype(int)
    
    correlations = []
    for col in numeric_cols:
        if col != 'loan_status':
            corr = np.corrcoef(df[col], loan_approved)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, abs(corr), corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    for col, abs_corr, corr in correlations[:10]:
        direction = "positive" if corr > 0 else "negative"
        print(f"   {col}: {corr:.3f} ({direction})")
    
    print("\n✅ Dataset generation completed successfully!")
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    dataset = main()
