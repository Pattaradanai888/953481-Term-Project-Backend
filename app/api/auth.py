from flask import request, jsonify
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    get_jwt_identity, jwt_required
)
from . import auth_bp
from ..models.user import User
from ..extensions import db
from datetime import datetime
import re


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()

    # Validate required fields
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    # Validate email format
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not re.match(email_regex, data['email']):
        return jsonify({'error': 'Invalid email format'}), 400

    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username is already taken'}), 409

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email is already registered'}), 409

    # Create new user
    new_user = User(
        username=data['username'],
        email=data['email'],
        created_at=datetime.utcnow()
    )
    new_user.set_password(data['password'])

    db.session.add(new_user)
    db.session.commit()

    # Generate tokens
    access_token = create_access_token(identity=new_user.user_id)
    refresh_token = create_refresh_token(identity=new_user.user_id)

    return jsonify({
        'message': 'User registered successfully',
        'user': new_user.to_dict(),
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login a user"""
    data = request.get_json()

    # Check for required fields
    if 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password are required'}), 400

    # Find user by username
    user = User.query.filter_by(username=data['username']).first()

    # Verify user and password
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid username or password'}), 401

    # Update last login time
    user.last_login = datetime.utcnow()
    db.session.commit()

    # Generate tokens
    access_token = create_access_token(identity=str(user.user_id))
    refresh_token = create_refresh_token(identity=str(user.user_id))

    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout a user"""
    # For JWT, logout is handled client-side by removing tokens
    # This endpoint exists for API consistency
    return jsonify({'message': 'Logout successful'}), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_me():
    """Get current user information"""
    try:
        user_id = get_jwt_identity()
        print("JWT Identity extracted:", repr(user_id))
    except Exception as e:
        print("JWT Error:", repr(e))
        return jsonify({"error": "Invalid token"}), 401

    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": user.to_dict()}), 200



@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=str(current_user))

    return jsonify({
        'access_token': new_access_token
    }), 200