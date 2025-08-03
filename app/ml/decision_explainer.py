from typing import Dict, Any, List
import json

class LoanDecisionExplainer:
    """Generate human-readable explanations for loan decisions"""
    
    def __init__(self):
        self.risk_thresholds = {
            'very_low': 20,
            'low': 40,
            'medium': 60,
            'high': 80,
            'very_high': 100
        }
        
        self.confidence_thresholds = {
            'very_high': 0.9,
            'high': 0.8,
            'medium': 0.7,
            'low': 0.6
        }
    
    def generate_explanation(self, 
                           loan_data: Dict[str, Any], 
                           prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive human-readable explanation"""
        
        approved = prediction_result['approved']
        confidence = prediction_result['confidence_score']
        risk_score = prediction_result['risk_score']
        conditions = prediction_result.get('conditions', [])
        risk_factors = prediction_result['prediction_details'].get('risk_factors', [])
        positive_factors = prediction_result['prediction_details'].get('positive_factors', [])
        
        # Generate main decision summary
        decision_summary = self._generate_decision_summary(approved, confidence, risk_score)
        
        # Generate detailed explanation
        detailed_explanation = self._generate_detailed_explanation(
            loan_data, approved, confidence, risk_score, risk_factors, positive_factors
        )
        
        # Generate risk assessment explanation
        risk_explanation = self._generate_risk_explanation(risk_score, risk_factors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            loan_data, approved, risk_score, conditions
        )
        
        # Generate next steps
        next_steps = self._generate_next_steps(approved, risk_score, conditions)
        
        return {
            'decision_summary': decision_summary,
            'detailed_explanation': detailed_explanation,
            'risk_assessment': risk_explanation,
            'key_factors': {
                'positive_factors': positive_factors,
                'risk_factors': risk_factors,
                'factor_explanations': self._explain_factors(risk_factors, positive_factors)
            },
            'recommendations': recommendations,
            'next_steps': next_steps,
            'explanation_metadata': {
                'confidence_level': self._get_confidence_level(confidence),
                'risk_category': self._get_risk_category(risk_score),
                'decision_basis': self._get_decision_basis(approved, confidence, risk_score)
            }
        }
    
    def _generate_decision_summary(self, approved: bool, confidence: float, risk_score: float) -> str:
        """Generate a concise decision summary"""
        
        confidence_level = self._get_confidence_level(confidence)
        risk_category = self._get_risk_category(risk_score)
        
        if approved:
            if confidence >= 0.8:
                summary = f"✅ **LOAN APPROVED** with {confidence_level} confidence. "
            else:
                summary = f"✅ **LOAN APPROVED** with {confidence_level} confidence. "
            
            if risk_score <= 40:
                summary += f"This is a {risk_category} risk application with favorable terms."
            elif risk_score <= 60:
                summary += f"This is a {risk_category} risk application requiring standard monitoring."
            else:
                summary += f"This is a {risk_category} risk application requiring enhanced oversight."
        else:
            summary = f"❌ **LOAN REJECTED** with {confidence_level} confidence. "
            summary += f"The application presents {risk_category} risk that exceeds our lending criteria."
        
        return summary
    
    def _generate_detailed_explanation(self, 
                                     loan_data: Dict[str, Any], 
                                     approved: bool, 
                                     confidence: float, 
                                     risk_score: float,
                                     risk_factors: List[str],
                                     positive_factors: List[str]) -> str:
        """Generate detailed explanation of the decision"""
        
        applicant_name = loan_data.get('name', 'The applicant')
        loan_amount = loan_data.get('loan_amount', 0)
        loan_purpose = loan_data.get('loan_purpose', 'unspecified')
        
        explanation = f"Our AI-powered assessment has evaluated {applicant_name}'s application for a ₹{loan_amount:,.0f} {loan_purpose.lower()} loan. "
        
        if approved:
            explanation += "The analysis indicates this application meets our lending criteria. "
            
            # Explain why it was approved
            if len(positive_factors) > 0:
                explanation += f"Key strengths include: {', '.join(positive_factors[:3])}. "
            
            if risk_score <= 30:
                explanation += "The applicant demonstrates excellent creditworthiness with minimal risk indicators. "
            elif risk_score <= 50:
                explanation += "The applicant shows good creditworthiness with manageable risk factors. "
            else:
                explanation += "While some risk factors are present, they are outweighed by positive indicators. "
                if len(risk_factors) > 0:
                    explanation += f"Areas requiring attention: {', '.join(risk_factors[:2])}. "
        
        else:
            explanation += "Unfortunately, the assessment reveals significant concerns that prevent approval. "
            
            if len(risk_factors) > 0:
                explanation += f"Primary concerns include: {', '.join(risk_factors[:3])}. "
            
            if risk_score > 80:
                explanation += "The high risk score indicates substantial likelihood of repayment difficulties. "
            elif risk_score > 60:
                explanation += "The elevated risk profile exceeds our current risk appetite for this loan type. "
            
            explanation += "We encourage addressing these concerns and reapplying in the future. "
        
        return explanation
    
    def _generate_risk_explanation(self, risk_score: float, risk_factors: List[str]) -> Dict[str, Any]:
        """Generate detailed risk assessment explanation"""
        
        risk_category = self._get_risk_category(risk_score)
        
        explanation = {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'risk_explanation': '',
            'risk_breakdown': {}
        }
        
        # Main risk explanation
        if risk_score <= 20:
            explanation['risk_explanation'] = "Excellent risk profile. The applicant demonstrates strong financial stability with minimal probability of default."
        elif risk_score <= 40:
            explanation['risk_explanation'] = "Good risk profile. The applicant shows solid financial health with low probability of repayment issues."
        elif risk_score <= 60:
            explanation['risk_explanation'] = "Moderate risk profile. Some concerns are present but manageable with appropriate monitoring."
        elif risk_score <= 80:
            explanation['risk_explanation'] = "Elevated risk profile. Multiple risk factors indicate higher probability of repayment challenges."
        else:
            explanation['risk_explanation'] = "High risk profile. Significant concerns suggest substantial probability of default or repayment difficulties."
        
        # Risk factor breakdown
        if risk_factors:
            explanation['risk_breakdown'] = {
                'identified_risks': risk_factors,
                'risk_impact': self._assess_risk_impact(risk_factors),
                'mitigation_suggestions': self._suggest_risk_mitigation(risk_factors)
            }
        
        return explanation
    
    def _generate_recommendations(self, 
                                loan_data: Dict[str, Any], 
                                approved: bool, 
                                risk_score: float, 
                                conditions: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if approved:
            # Recommendations for approved loans
            if risk_score > 60:
                recommendations.append("Consider setting up automatic EMI payments to ensure timely repayments")
                recommendations.append("Maintain regular income documentation for periodic reviews")
                
            if loan_data.get('collateral_value', 0) == 0 and loan_data.get('loan_amount', 0) > 500000:
                recommendations.append("Consider providing collateral to secure better interest rates")
            
            if not loan_data.get('has_life_insurance', False):
                recommendations.append("Obtain life insurance to protect against unforeseen circumstances")
            
            recommendations.append("Build emergency fund equivalent to 6 months of EMI payments")
            recommendations.append("Monitor credit score regularly and maintain good payment history")
            
        else:
            # Recommendations for rejected loans
            credit_score = loan_data.get('credit_score', 650)
            if credit_score < 700:
                recommendations.append(f"Improve credit score from {credit_score} to above 700 through timely payments")
            
            total_income = loan_data.get('applicant_income', 0) + loan_data.get('coapplicant_income', 0)
            loan_amount = loan_data.get('loan_amount', 0)
            if total_income > 0 and (loan_amount / total_income) > 4:
                recommendations.append("Consider applying for a lower loan amount or increase household income")
            
            if loan_data.get('loan_default_history', 0) > 0:
                recommendations.append("Clear any existing defaults and maintain clean credit history for 12+ months")
            
            if loan_data.get('years_in_current_job', 0) < 2:
                recommendations.append("Build employment stability (2+ years in current role) before reapplying")
            
            recommendations.append("Reduce existing EMI obligations to improve debt-to-income ratio")
            recommendations.append("Consider adding a co-applicant with strong credit profile")
            recommendations.append("Build savings and consider providing collateral for future applications")
        
        return recommendations
    
    def _generate_next_steps(self, approved: bool, risk_score: float, conditions: List[str]) -> List[str]:
        """Generate clear next steps for the applicant"""
        
        next_steps = []
        
        if approved:
            next_steps.append("🎉 Congratulations! Your loan has been approved")
            next_steps.append("📋 Review the loan terms and conditions carefully")
            
            if conditions:
                next_steps.append("✅ Complete the following requirements:")
                for condition in conditions:
                    next_steps.append(f"   • {condition}")
            
            next_steps.append("📝 Submit required documentation (as per conditions)")
            next_steps.append("🏦 Schedule loan agreement signing and disbursement")
            next_steps.append("📱 Set up EMI auto-debit for hassle-free repayments")
            
        else:
            next_steps.append("📞 Contact our relationship manager for detailed feedback")
            next_steps.append("📊 Review the specific areas of concern mentioned above")
            next_steps.append("🎯 Work on improving the highlighted risk factors")
            next_steps.append("⏳ Wait at least 3-6 months before reapplying")
            next_steps.append("💡 Consider our financial counseling services")
            next_steps.append("📈 Monitor your progress and reapply when conditions improve")
        
        return next_steps
    
    def _explain_factors(self, risk_factors: List[str], positive_factors: List[str]) -> Dict[str, str]:
        """Provide explanations for key factors"""
        
        factor_explanations = {}
        
        # Risk factor explanations
        risk_explanations = {
            "Poor credit score": "Credit scores below 600 indicate higher likelihood of payment defaults",
            "Previous loan defaults": "Past defaults suggest potential difficulty in meeting future payment obligations",
            "High loan-to-income ratio": "When loan amount exceeds 4-5 times annual income, repayment becomes challenging",
            "High debt-to-income ratio": "When existing debts exceed 60% of income, additional loans pose significant risk",
            "Short employment history": "Less than 2 years in current job indicates income instability",
            "High regional default rate": "Areas with high default rates present additional collection challenges",
            "Large loan without collateral": "Unsecured large loans carry higher risk for the lender"
        }
        
        # Positive factor explanations
        positive_explanations = {
            "Excellent credit score": "Credit scores above 750 demonstrate responsible credit behavior",
            "Stable employment": "Long-term employment indicates reliable income source",
            "Life insurance coverage": "Insurance provides protection against income loss scenarios",
            "Dual income household": "Multiple income sources reduce repayment risk",
            "Strong collateral coverage": "Adequate collateral secures the loan and reduces lender risk",
            "Good savings habit": "Regular savings demonstrate financial discipline and planning"
        }
        
        # Match and explain factors
        for factor in risk_factors:
            for key, explanation in risk_explanations.items():
                if key.lower() in factor.lower():
                    factor_explanations[factor] = explanation
                    break
        
        for factor in positive_factors:
            for key, explanation in positive_explanations.items():
                if key.lower() in factor.lower():
                    factor_explanations[factor] = explanation
                    break
        
        return factor_explanations
    
    def _assess_risk_impact(self, risk_factors: List[str]) -> Dict[str, str]:
        """Assess the impact level of each risk factor"""
        
        impact_assessment = {}
        
        high_impact_keywords = ['default', 'poor credit', 'high ratio']
        medium_impact_keywords = ['delay', 'short employment', 'regional']
        
        for factor in risk_factors:
            factor_lower = factor.lower()
            if any(keyword in factor_lower for keyword in high_impact_keywords):
                impact_assessment[factor] = "High Impact"
            elif any(keyword in factor_lower for keyword in medium_impact_keywords):
                impact_assessment[factor] = "Medium Impact"
            else:
                impact_assessment[factor] = "Low Impact"
        
        return impact_assessment
    
    def _suggest_risk_mitigation(self, risk_factors: List[str]) -> Dict[str, str]:
        """Suggest mitigation strategies for risk factors"""
        
        mitigation_suggestions = {}
        
        mitigation_mapping = {
            "credit score": "Pay all bills on time, reduce credit utilization, avoid new credit inquiries",
            "default": "Clear outstanding defaults, maintain clean payment history for 12+ months",
            "income ratio": "Increase income, reduce loan amount, or extend loan tenure",
            "employment": "Complete probation period, maintain stable job for 2+ years",
            "collateral": "Provide adequate security or consider secured loan options",
            "regional": "Consider providing additional guarantees or co-applicant from low-risk area"
        }
        
        for factor in risk_factors:
            factor_lower = factor.lower()
            for keyword, suggestion in mitigation_mapping.items():
                if keyword in factor_lower:
                    mitigation_suggestions[factor] = suggestion
                    break
        
        return mitigation_suggestions
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.6:
            return "moderate"
        else:
            return "low"
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to human-readable category"""
        if risk_score <= 20:
            return "very low"
        elif risk_score <= 40:
            return "low"
        elif risk_score <= 60:
            return "medium"
        elif risk_score <= 80:
            return "high"
        else:
            return "very high"
    
    def _get_decision_basis(self, approved: bool, confidence: float, risk_score: float) -> str:
        """Explain the basis for the decision"""
        if approved:
            if confidence >= 0.8 and risk_score <= 40:
                return "Strong financial profile with minimal risk indicators"
            elif confidence >= 0.7:
                return "Acceptable risk profile meeting lending criteria"
            else:
                return "Borderline approval based on overall assessment"
        else:
            if risk_score >= 80:
                return "High risk profile exceeding lending thresholds"
            elif confidence <= 0.6:
                return "Insufficient confidence in repayment capability"
            else:
                return "Risk factors outweigh positive indicators"
    
    def generate_plain_text_summary(self, explanation: Dict[str, Any]) -> str:
        """Generate a plain text summary for simple display"""
        
        summary = explanation['decision_summary'] + "\n\n"
        summary += explanation['detailed_explanation'] + "\n\n"
        
        if explanation['key_factors']['positive_factors']:
            summary += "Strengths:\n"
            for factor in explanation['key_factors']['positive_factors']:
                summary += f"• {factor}\n"
            summary += "\n"
        
        if explanation['key_factors']['risk_factors']:
            summary += "Areas of Concern:\n"
            for factor in explanation['key_factors']['risk_factors']:
                summary += f"• {factor}\n"
            summary += "\n"
        
        if explanation['recommendations']:
            summary += "Recommendations:\n"
            for rec in explanation['recommendations'][:3]:  # Top 3 recommendations
                summary += f"• {rec}\n"
        
        return summary