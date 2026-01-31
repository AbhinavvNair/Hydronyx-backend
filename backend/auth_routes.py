from fastapi import APIRouter, HTTPException, status, Header, Body
from datetime import timedelta, datetime
from bson.objectid import ObjectId
from pydantic import BaseModel, EmailStr
from auth_utils import (
    UserRegister, UserLogin, Token, UserResponse,
    hash_password, verify_password, create_access_token,
    create_refresh_token, verify_token, ACCESS_TOKEN_EXPIRE_MINUTES
)
from database import get_users_collection
from email_service import send_verification_email, send_password_reset_email

router = APIRouter(prefix="/api/auth", tags=["authentication"])


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    """Register a new user"""
    users_collection = get_users_collection()
    
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = hash_password(user.password)
    user_doc = {
        "email": user.email,
        "name": user.name,
        "password": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True,
        "is_verified": False,
        "verification_token": None,
        "role": "viewer",  # RBAC: admin, analyst, viewer
    }
    
    result = users_collection.insert_one(user_doc)
    user_doc["id"] = str(result.inserted_id)
    from auth_utils import create_access_token
    token = create_access_token({"sub": user.email, "type": "verify"}, expires_delta=timedelta(hours=24))
    users_collection.update_one({"email": user.email}, {"$set": {"verification_token": token}})
    send_verification_email(user.email, token, user.name)
    
    return UserResponse(
        id=user_doc["id"],
        email=user_doc["email"],
        name=user_doc["name"],
        created_at=user_doc["created_at"]
    )



@router.get("/verify-email")
async def verify_email(token: str):
    """Verify an email using the token sent at registration.

    In production this should be linked from an email; here it sets `is_verified`.
    """
    users_collection = get_users_collection()
    # find user with matching token
    user = users_collection.find_one({"verification_token": token})
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    users_collection.update_one({"_id": user["_id"]}, {"$set": {"is_verified": True, "verification_token": None}})
    return {"status": "ok", "message": "Email verified"}


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user and return tokens"""
    users_collection = get_users_collection()
    
    # Find user by email
    user = users_collection.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password
    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user["email"]})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh(refresh_token: str):
    """Refresh access token using refresh token"""
    email = verify_token(refresh_token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    users_collection = get_users_collection()
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user(authorization: str = Header(None)):
    """Get current user profile"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    # Extract token from "Bearer <token>"
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    token = authorization.split(" ")[1]
    email = verify_token(token)
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    users_collection = get_users_collection()
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        name=user["name"],
        created_at=user["created_at"],
        role=user.get("role", "viewer"),
    )


@router.post("/logout")
async def logout():
    """Logout user (token invalidation handled by client)"""
    return {"message": "Logged out successfully"}


@router.post("/forgot-password")
async def forgot_password(body: PasswordResetRequest):
    """Request password reset. Sends email with reset link if user exists."""
    users_collection = get_users_collection()
    user = users_collection.find_one({"email": body.email})
    if not user:
        return {"status": "ok", "message": "If an account exists, a reset link was sent."}
    from auth_utils import create_access_token
    token = create_access_token({"sub": user["email"], "type": "reset"}, expires_delta=timedelta(hours=1))
    users_collection.update_one({"email": body.email}, {"$set": {"reset_token": token, "reset_token_expires": datetime.utcnow() + timedelta(hours=1)}})
    send_password_reset_email(body.email, token)
    return {"status": "ok", "message": "If an account exists, a reset link was sent."}


@router.post("/reset-password")
async def reset_password(body: PasswordResetConfirm):
    """Confirm password reset using token from email."""
    users_collection = get_users_collection()
    user = users_collection.find_one({"reset_token": body.token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    expires = user.get("reset_token_expires")
    if expires and datetime.utcnow() > expires:
        users_collection.update_one({"_id": user["_id"]}, {"$unset": {"reset_token": "", "reset_token_expires": ""}})
        raise HTTPException(status_code=400, detail="Reset token has expired")
    hashed = hash_password(body.new_password)
    users_collection.update_one({"_id": user["_id"]}, {"$set": {"password": hashed}, "$unset": {"reset_token": "", "reset_token_expires": ""}})
    return {"status": "ok", "message": "Password reset successfully"}


@router.post("/rotate-secret")
async def rotate_secret(new_secret: str, authorization: str = Header(None)):
    """Rotate the JWT secret in-memory. Caller must be an admin.

    NOTE: This only updates the running process; persist new secret in environment
    or secret manager for production deployments.
    """
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    users_collection = get_users_collection()
    user = users_collection.find_one({"email": email})
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    # RBAC: require admin role or is_admin flag
    if user.get("role") != "admin" and not user.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")

    from auth_utils import rotate_secret as _rotate
    _rotate(new_secret)
    return {"status": "ok", "message": "Secret rotated in-memory; persist externally"}
