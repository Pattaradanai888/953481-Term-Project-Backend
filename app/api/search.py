from flask import Blueprint, request, jsonify, current_app
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

    # Use SymSpell for phrase-level correction
    suggestions = current_app.sym_spell.lookup_compound(query, max_edit_distance=2)
    best_suggestion = suggestions[0] if suggestions else None
    corrected_query = best_suggestion.term if best_suggestion and best_suggestion.distance > 0 else None
    search_term = corrected_query or query  # Use corrected term if available, else original query

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    offset = (page - 1) * per_page

    # Full-text search using PostgreSQL tsvector
    search_query = func.plainto_tsquery('english', search_term)
    base_query = Recipe.query.filter(
        Recipe.search_vector.op('@@')(search_query)
    ).order_by(
        func.ts_rank(Recipe.search_vector, search_query).desc()
    )

    total_results = base_query.count()
    results = base_query.offset(offset).limit(per_page).all()

    # Fallback for no results and no prior correction
    if not results and not corrected_query:
        similar = Recipe.query.filter(
            func.similarity(Recipe.name, query) > 0.3
        ).order_by(func.similarity(Recipe.name, query).desc()).first()
        if similar:
            corrected_query = similar.name

    response = {
        'results': [recipe.to_dict() for recipe in results],
        'total': total_results,
        'page': page,
        'per_page': per_page,
        'correctedQuery': corrected_query,  # None if no correction was made
    }
    return jsonify(response), 200

@search_bp.route('/recipe/<int:recipe_id>', methods=['GET'], strict_slashes=False)
@jwt_required()
def get_recipe(recipe_id):
    recipe = Recipe.query.get_or_404(recipe_id)
    return jsonify(recipe.to_dict()), 200