from datetime import datetime
from flask_jwt_extended import create_access_token, create_refresh_token
import re
from ..models.user import User
from ..extensions import db


class AuthService:
    """Service class for handling authentication logic"""

    @staticmethod
    def validate_email(email):
        """Validate email format"""
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(email_regex, email) is not None

    @staticmethod
    def get_user_by_username(username):
        """Get user by username"""
        return User.query.filter_by(username=username).first()

    @staticmethod
    def get_user_by_email(email):
        """Get user by email"""
        return User.query.filter_by(email=email).first()

    @staticmethod
    def get_user_by_id(user_id):
        """Get user by ID"""
        return User.query.get(user_id)

    @classmethod
    def register_user(cls, username, email, password):
        """Register a new user"""
        # Validate email format
        if not cls.validate_email(email):
            return {"success": False, "error": "Invalid email format"}, 400

        # Check if username already exists
        if cls.get_user_by_username(username):
            return {"success": False, "error": "Username is already taken"}, 409

        # Check if email already exists
        if cls.get_user_by_email(email):
            return {"success": False, "error": "Email is already registered"}, 409

        # Create new user
        new_user = User(
            username=username,
            email=email,
            created_at=datetime.utcnow()
        )
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        # Generate tokens
        access_token = create_access_token(identity=str(new_user.user_id))
        refresh_token = create_refresh_token(identity=str(new_user.user_id))

        return {
            "success": True,
            "message": "User registered successfully",
            "user": new_user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token
        }, 201

    @classmethod
    def login_user(cls, username, password):
        """Login a user"""
        # Find user by username
        user = cls.get_user_by_username(username)

        # Verify user and password
        if not user or not user.check_password(password):
            return {"success": False, "error": "Invalid username or password"}, 401

        # Update last login time
        user.last_login = datetime.utcnow()
        db.session.commit()

        # Generate tokens
        access_token = create_access_token(str(user.user_id))
        refresh_token = create_refresh_token(str(user.user_id))

        return {
            "success": True,
            "message": "Login successful",
            "user": user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token
        }, 200

    @staticmethod
    def refresh_token(user_id):
        """Generate a new access token"""
        new_access_token = create_access_token(identity=str(user_id))

        return {
            "success": True,
            "access_token": new_access_token
        }, 200