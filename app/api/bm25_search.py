from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required
import os
from app.extensions import db

from . import bm25_search_bp
from app.utils.bm25 import BM25RecipeSearch

# Initialize BM25 search instance
bm25_searcher = None

def initialize_bm25_search():
    """Initialize BM25 search once when the app starts"""
    global bm25_searcher

    # Prevent reinitialization if already set
    if bm25_searcher is not None:
        return

    # Ensure we are within an application context
    with current_app.app_context():
        # Get processed data path from app config or use default
        processed_data_path = current_app.config.get(
            'PROCESSED_RECIPES_PATH',
            "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/recipes_processed.csv"
        )

        # Check if file exists
        if not os.path.exists(processed_data_path):
            current_app.logger.error(f"Processed recipes file not found: {processed_data_path}")
            return

        # Initialize BM25 search
        bm25_searcher = BM25RecipeSearch(processed_data_path)
        current_app.logger.info("BM25 search initialized successfully")


@bm25_search_bp.route('/bm25search', methods=['GET'], strict_slashes=False)
@jwt_required()
def search_recipes_bm25():
    """Search recipes using BM25 with spell correction"""
    global bm25_searcher

    # Ensure BM25 is initialized
    if bm25_searcher is None:
        initialize_bm25_search()

    # Get query parameter and validate
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

    # Check if BM25 search is initialized
    if not bm25_searcher:
        return jsonify({
            'error': 'BM25 search not initialized',
            'results': [],
            'total': 0,
            'page': page,
            'per_page': per_page
        }), 500

    # Perform BM25 search using the search term
    search_results = bm25_searcher.search(
        query=search_term,
        page=page,
        per_page=per_page
    )

    # Convert to database models for consistent output format
    from app.models.recipe import Recipe

    # Get recipe IDs from search results
    recipe_ids = [int(result['RecipeId']) for result in search_results['results']
                  if 'RecipeId' in result and result['RecipeId']]

    # Fetch recipes from database in the same order
    db_recipes = []
    for recipe_id in recipe_ids:
        recipe = Recipe.query.get(recipe_id)
        if recipe:
            db_recipes.append(recipe)

    # Prepare response including the original query and any correction made
    response = {
        'results': [recipe.to_dict() for recipe in db_recipes],
        'total': search_results['total'],
        'page': page,
        'per_page': per_page,
        'query': query,
        'correctedQuery': corrected_query  # Will be None if no correction was applied
    }

    return jsonify(response), 200


def register_bm25(app):
    """Register BM25 initialization when the app starts"""
    app.before_first_request(initialize_bm25_search)
