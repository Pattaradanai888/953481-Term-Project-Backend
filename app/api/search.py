from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.models.recipe import Recipe
from app.extensions import db
from sqlalchemy import func

search_bp = Blueprint('search', __name__)

@search_bp.route('/search', methods=['GET'])
@jwt_required()
def search_recipes():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Missing search query'}), 400

    # Full-text search using PostgreSQL tsvector
    search_query = func.plainto_tsquery('english', query)
    results = Recipe.query.filter(
        Recipe.search_vector.op('@@')(search_query)
    ).order_by(
        func.ts_rank(Recipe.search_vector, search_query).desc()
    ).limit(20).all()

    # Simple typo correction (placeholder)
    corrected_query = None
    if not results and ' ' not in query:  # Basic check for potential typos
        # This is a simple example; use pg_trgm or a spell-check library for better results
        similar = Recipe.query.filter(
            func.similarity(Recipe.name, query) > 0.3
        ).order_by(func.similarity(Recipe.name, query).desc()).first()
        corrected_query = similar.name if similar else None

    response = {
        'results': [recipe.to_dict() for recipe in results],
        'correctedQuery': corrected_query,
    }
    return jsonify(response), 200