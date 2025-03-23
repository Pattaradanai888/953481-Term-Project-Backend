from flask import Blueprint, request, jsonify, url_for
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.folder import Folder
from app.models.bookmark import Bookmark
from app.extensions import db
import os
from werkzeug.utils import secure_filename
from . import folder_bp
from ..models import Recipe
from ..services import RecommendServiceOption2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'static/uploads'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@folder_bp.route('', methods=['GET'])
@jwt_required()
def get_folders():
    user_id = get_jwt_identity()
    folders = Folder.query.filter_by(user_id=user_id).all()
    return jsonify([folder.to_dict() for folder in folders]), 200


@folder_bp.route('', methods=['POST'])
@jwt_required()
def create_folder():
    user_id = get_jwt_identity()
    data = request.form
    name = data.get('name')
    description = data.get('description')
    cover_file = request.files.get('cover')

    if not name:
        return jsonify({'error': 'Folder name is required'}), 400

    folder = Folder(user_id=user_id, name=name, description=description)

    if cover_file and allowed_file(cover_file.filename):
        filename = secure_filename(cover_file.filename)
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        upload_dir = os.path.join(app_root, UPLOAD_FOLDER)
        cover_path = os.path.join(upload_dir, filename)
        os.makedirs(upload_dir, exist_ok=True)
        cover_file.save(cover_path)
        folder.cover_url = url_for('static', filename=f'uploads/{filename}', _external=True)

    db.session.add(folder)
    db.session.commit()
    return jsonify(folder.to_dict()), 201


@folder_bp.route('/<int:folder_id>', methods=['GET'])
@jwt_required()
def get_folder(folder_id):
    user_id = get_jwt_identity()
    folder = Folder.query.filter_by(folder_id=folder_id, user_id=user_id).first_or_404()
    bookmarks = Bookmark.query.filter_by(folder_id=folder_id, user_id=user_id).all()

    recipes = []
    for b in bookmarks:
        recipe = Recipe.query.filter_by(recipe_id=b.recipe_id).first()
        if recipe:
            recipe_dict = recipe.to_dict()
            recipes.append({
                'recipe_id': recipe.recipe_id,
                'name': recipe.name,
                'description': recipe.description,
                'image_url': recipe_dict.get('image_url'),  # Using recipe image_url
                'user_rating': b.user_rating
            })
        else:
            # If recipe isn't found, you might want to handle it differently
            recipes.append({
                'recipe_id': b.recipe_id,
                'name': f'Recipe {b.recipe_id}',
                'description': 'Recipe not found',
                'image_url': None,
                'user_rating': b.user_rating,
            })

    return jsonify({
        'folder': folder.to_dict(),
        'recipes': recipes
    }), 200


@folder_bp.route('/<int:folder_id>', methods=['PUT'])
@jwt_required()
def update_folder(folder_id):
    user_id = get_jwt_identity()
    folder = Folder.query.filter_by(folder_id=folder_id, user_id=user_id).first_or_404()
    data = request.form
    name = data.get('name')
    description = data.get('description')
    cover_file = request.files.get('cover')

    if name:
        folder.name = name
    if description:
        folder.description = description
    if cover_file and allowed_file(cover_file.filename):
        filename = secure_filename(cover_file.filename)
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        upload_dir = os.path.join(app_root, UPLOAD_FOLDER)
        cover_path = os.path.join(upload_dir, filename)
        os.makedirs(upload_dir, exist_ok=True)
        cover_file.save(cover_path)
        folder.cover_url = url_for('static', filename=f'uploads/{filename}', _external=True)

    db.session.commit()
    return jsonify(folder.to_dict()), 200


@folder_bp.route('/<int:folder_id>', methods=['DELETE'])
@jwt_required()
def delete_folder(folder_id):
    user_id = get_jwt_identity()
    folder = Folder.query.filter_by(folder_id=folder_id, user_id=user_id).first_or_404()
    db.session.delete(folder)
    db.session.commit()
    return jsonify({'message': 'Folder deleted successfully'}), 200


@folder_bp.route("/<int:folder_id>/suggestions", methods=["GET"])
@jwt_required()
def get_folder_suggestions(folder_id):
    user_id = get_jwt_identity()
    folder = Folder.query.filter_by(folder_id=folder_id, user_id=user_id).first()
    if not folder:
        return jsonify({"error": "Folder not found or not accessible."}), 404

    recommendations = RecommendServiceOption2.generate_folder_suggestions(user_id, folder_id, top_k=10)
    return jsonify({"suggestions": recommendations}), 200