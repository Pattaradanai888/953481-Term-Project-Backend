from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.models.recipe import Recipe
from app.extensions import db
from sqlalchemy import func
from . import search_bp

@search_bp.route('/search', methods=['GET'], strict_slashes=False)
@jwt_required()
def search_recipes():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Missing search query'}), 400

    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    offset = (page - 1) * per_page

    # Full-text search using PostgreSQL tsvector
    search_query = func.plainto_tsquery('english', query)
    base_query = Recipe.query.filter(
        Recipe.search_vector.op('@@')(search_query)
    ).order_by(
        func.ts_rank(Recipe.search_vector, search_query).desc()
    )

    total_results = base_query.count()
    results = base_query.offset(offset).limit(per_page).all()

    # Simple typo correction (placeholder)
    corrected_query = None
    if not results and ' ' not in query:  # Basic check for potential typos
        similar = Recipe.query.filter(
            func.similarity(Recipe.name, query) > 0.3
        ).order_by(func.similarity(Recipe.name, query).desc()).first()
        corrected_query = similar.name if similar else None

    response = {
        'results': [recipe.to_dict() for recipe in results],
        'total': total_results,
        'page': page,
        'per_page': per_page,
        'correctedQuery': corrected_query,
    }
    return jsonify(response), 200

@search_bp.route('/recipe/<int:recipe_id>', methods=['GET'], strict_slashes=False)
@jwt_required()
def get_recipe(recipe_id):
    recipe = Recipe.query.get_or_404(recipe_id)
    return jsonify(recipe.to_dict()), 200
