import pytest
from fastapi.testclient import TestClient

def test_create_loan_application(client: TestClient):
    loan_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "gender": "Male",
        "married": "Yes",
        "dependents": 2,
        "education": "Graduate",
        "age": 35,
        "applicant_income": 75000,
        "coapplicant_income": 25000,
        "monthly_expenses": 15000,
        "loan_amount": 300000,
        "loan_amount_term": 360,
        "credit_score": 720,
        "credit_history": 1,
        "property_area": "Urban"
    }
    
    response = client.post("/loans/", json=loan_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "id" in data
    assert "prediction_result" in data
    assert data["status"] == "processed"

def test_get_loan_applications(client: TestClient, rm_token: str):
    headers = {"Authorization": f"Bearer {rm_token}"}
    response = client.get("/loans/", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)

def test_loan_application_validation(client: TestClient):
    # Test with invalid data
    invalid_data = {
        "name": "J",  # Too short
        "email": "invalid-email",  # Invalid format
        "age": 17,  # Below minimum
        "applicant_income": -1000  # Negative income
    }
    
    response = client.post("/loans/", json=invalid_data)
    assert response.status_code == 422