from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.bookmark import Bookmark  # Assume a Bookmark model

from . import bookmark_bp


@bookmark_bp.route('', methods=['POST'])
@jwt_required()
def create_bookmark():
    user_id = get_jwt_identity()
    data = request.get_json()
    recipe_id = data.get('recipe_id')
    folder_id = data.get('folder_id')
    user_rating = data.get('rating')

    if not all([recipe_id, folder_id, user_rating]) or user_rating not in range(1, 6):
        return jsonify({'error': 'Missing or invalid data'}), 400

    # Check if bookmark already exists
    existing_bookmark = Bookmark.query.filter_by(
        user_id=user_id,
        recipe_id=recipe_id,
        folder_id=folder_id
    ).first()

    if existing_bookmark:
        # Optionally update the rating if it exists
        existing_bookmark.user_rating = user_rating
        db.session.commit()
        return jsonify({
            'message': 'Bookmark updated',
            'bookmark': existing_bookmark.to_dict()
        }), 200

    # Create new bookmark if it doesn’t exist
    bookmark = Bookmark(
        user_id=user_id,
        recipe_id=recipe_id,
        folder_id=folder_id,
        user_rating=user_rating
    )
    db.session.add(bookmark)
    db.session.commit()
    return jsonify({'message': 'Bookmark created', 'bookmark': bookmark.to_dict()}), 201