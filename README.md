# 1. Setup Instructions

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Set up PostgreSQL database
4. Configure environment variables in .env file:
   DATABASE_URL=postgresql://user:password@localhost:5432/loan_db
   SECRET_KEY=your-secret-key-here
   ACCESS_TOKEN_EXPIRE_MINUTES=30
5. Run migrations: alembic upgrade head
6. Create initial superadmin user
7. Start the application: uvicorn app.main:app --reload

# 2. Complete API Documentation

## Authentication
POST /auth/login - Login with username/password
POST /auth/login-json - Login with JSON payload
POST /auth/refresh - Refresh access token
GET /auth/me - Get current user info

## User Management (Superadmin only)
POST /users/ - Create new user
GET /users/ - Get all users
GET /users/{id} - Get specific user
PUT /users/{id} - Update user
DELETE /users/{id} - Delete user
PATCH /users/{id}/disable - Disable user
PATCH /users/{id}/enable - Enable user

## Loan Applications
POST /loans/ - Create loan application
GET /loans/ - Get paginated applications with filters
GET /loans/{id} - Get specific application

## Weight Management (Superadmin only)
GET /weights/ - Get all field weights
PUT /weights/ - Update single field weight
PUT /weights/bulk - Bulk update weights
POST /weights/initialize - Initialize default weights

## Model Management (Superadmin only)
POST /model/train - Train new model with data
GET /model/performance - Get model performance
POST /model/reload - Reload model from disk

## Reports & Analytics
GET /reports/dashboard - Dashboard statistics
GET /reports/trends - Approval trends over time
GET /reports/risk-analysis - Detailed risk analysis

# 3. Example API Calls

## Login
curl -X POST "http://localhost:8000/auth/login" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "username=admin&password=admin123"

## Create Loan Application
curl -X POST "http://localhost:8000/loans/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN" \
-d '{
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
}'

## Update Field Weight
curl -X PUT "http://localhost:8000/weights/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN" \
-d '{
  "field_name": "credit_score",
  "weight": 2.5
}'

## Get Dashboard Stats
curl -X GET "http://localhost:8000/reports/dashboard?days=30" \
-H "Authorization: Bearer YOUR_TOKEN"

# 4. Key Features Implemented

✅ Complete loan application processing with ML prediction
✅ Configurable field weights system with bulk updates
✅ User management (Superadmin/RM roles) with full CRUD
✅ Comprehensive data preprocessing and feature engineering
✅ Risk scoring and detailed prediction analysis
✅ RESTful API with pagination and filtering
✅ Clean architecture following SOLID principles
✅ Comprehensive error handling and validation
✅ Docker support for easy deployment
✅ Model training and management endpoints
✅ Dashboard and reporting system
✅ Advanced validators and helpers
✅ Bulk operations support
✅ Trend analysis and risk analytics

# 5. Security Features

✅ JWT-based authentication with refresh tokens
✅ Role-based access control (Superadmin/RM)
✅ Password strength validation
✅ Input validation and sanitization
✅ SQL injection protection
✅ Sensitive data masking in logs
✅ Comprehensive exception handling

# 6. Production Deployment

The system is production-ready with:
- Docker containerization
- Environment-based configuration
- Comprehensive logging
- Error handling and monitoring
- Database migrations
- Model versioning support
- Scalable architecture

# 7. Database Setup Scripts

# alembic.ini
"""
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://user:password@localhost:5432/loan_db

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""


 python scripts/train_models.py --models xgboost random_forest gradient_boosting logistic_regression