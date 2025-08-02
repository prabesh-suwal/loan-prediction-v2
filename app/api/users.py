from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from ..config.database import get_db
from ..services.user_service import UserService
from ..schemas.user import UserCreate, UserUpdate, UserResponse
from ..core.dependencies import get_current_user, get_current_superadmin

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Create new user (superadmin only)"""
    
    user_service = UserService(db)
    
    try:
        new_user = user_service.create_user(user_data, current_user.role)
        return UserResponse.from_orm(new_user)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Get all users (superadmin only)"""
    
    user_service = UserService(db)
    users = user_service.get_all_users(skip=skip, limit=limit)
    
    return [UserResponse.from_orm(user) for user in users]

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Get specific user (superadmin only)"""
    
    user_service = UserService(db)
    user = user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse.from_orm(user)

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Update user (superadmin only)"""
    
    user_service = UserService(db)
    
    try:
        updated_user = user_service.update_user(user_id, user_data, current_user.role)
        
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse.from_orm(updated_user)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Delete user (superadmin only)"""
    
    user_service = UserService(db)
    
    try:
        success = user_service.delete_user(user_id, current_user.role)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@router.patch("/{user_id}/disable")
async def disable_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Disable user (superadmin only)"""
    
    user_service = UserService(db)
    
    try:
        disabled_user = user_service.disable_user(user_id, current_user.role)
        
        if not disabled_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User disabled successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error disabling user: {str(e)}")

@router.patch("/{user_id}/enable")
async def enable_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_superadmin)
):
    """Enable user (superadmin only)"""
    
    user_service = UserService(db)
    
    try:
        enabled_user = user_service.enable_user(user_id, current_user.role)
        
        if not enabled_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User enabled successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enabling user: {str(e)}")