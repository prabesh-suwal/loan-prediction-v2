from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..repositories.user_repository import UserRepository
from ..schemas.user import UserCreate, UserUpdate
from ..models.user import User
from ..core.security import get_password_hash, verify_password
from ..utils.validators import validate_email, validate_password_strength

class UserService:
    def __init__(self, db: Session):
        self.db = db
        self.user_repository = UserRepository(db)
    
    def create_user(self, user_data: UserCreate, created_by_role: str) -> User:
        """Create new user (only superadmin can create users)"""
        
        if created_by_role != "superadmin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superadmin can create users"
            )
        
        # Validate email format
        if not validate_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Validate password strength
        is_strong, message = validate_password_strength(user_data.password)
        if not is_strong:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Check if username already exists
        existing_user = self.user_repository.get_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        existing_email = self.user_repository.get_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Create user
        user_dict = user_data.dict(exclude={'password'})
        user_dict['hashed_password'] = get_password_hash(user_data.password)
        
        return self.user_repository.create(user_dict)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        
        user = self.user_repository.get_by_username(username)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.user_repository.get_by_id(user_id)
    
    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users (superadmin only)"""
        return self.user_repository.get_all(skip=skip, limit=limit)
    
    def update_user(self, user_id: int, user_data: UserUpdate, updated_by_role: str) -> Optional[User]:
        """Update user (superadmin only)"""
        
        if updated_by_role != "superadmin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superadmin can update users"
            )
        
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        update_dict = user_data.dict(exclude_unset=True)
        
        # Hash password if provided
        if 'password' in update_dict:
            is_strong, message = validate_password_strength(update_dict['password'])
            if not is_strong:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=message
                )
            update_dict['hashed_password'] = get_password_hash(update_dict['password'])
            del update_dict['password']
        
        return self.user_repository.update(user_id, update_dict)
    
    def delete_user(self, user_id: int, deleted_by_role: str) -> bool:
        """Delete user (superadmin only)"""
        
        if deleted_by_role != "superadmin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superadmin can delete users"
            )
        
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Don't allow deleting the last superadmin
        if user.role == "superadmin":
            superadmin_count = len([u for u in self.user_repository.get_all() if u.role == "superadmin"])
            if superadmin_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the last superadmin"
                )
        
        return self.user_repository.delete(user_id)
    
    def disable_user(self, user_id: int, disabled_by_role: str) -> Optional[User]:
        """Disable user (superadmin only)"""
        
        if disabled_by_role != "superadmin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superadmin can disable users"
            )
        
        return self.user_repository.update(user_id, {'is_active': False})
    
    def enable_user(self, user_id: int, enabled_by_role: str) -> Optional[User]:
        """Enable user (superadmin only)"""
        
        if enabled_by_role != "superadmin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superadmin can enable users"
            )
        
        return self.user_repository.update(user_id, {'is_active': True})