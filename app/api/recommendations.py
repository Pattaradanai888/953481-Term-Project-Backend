# app/api/recommendations.py
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models import Recipe, Bookmark
from sqlalchemy import func, desc
from app.extensions import db, cache  # Make sure cache is configured
from . import recommendations_bp
from ..services.recommend_service import RecommendService


def make_recommendations_cache_key():
    # Get the user id from the JWT (it must be available because of jwt_required)
    user_id = get_jwt_identity() or ''
    # Get the folder_id from query parameters
    folder_id = request.args.get('folder_id', '')
    # Return a unique key for this endpoint call
    return f"recommendations_{user_id}_{folder_id}_{request.path}"

@recommendations_bp.route('/recommendations', methods=['GET'])
@jwt_required()
@cache.cached(timeout=300, key_prefix=make_recommendations_cache_key)
def get_recommendations():
    user_id = get_jwt_identity()
    folder_id = request.args.get('folder_id', type=int)

    # 1. Recommendations based on the user's bookmarks in the folder (if provided)
    bookmark_query = Bookmark.query.filter_by(user_id=user_id)
    if folder_id:
        bookmark_query = bookmark_query.filter_by(folder_id=folder_id)
    recent_bookmarks = bookmark_query.order_by(desc(Bookmark.created_at)).limit(9).all()
    bookmark_recipe_ids = [bookmark.recipe_id for bookmark in recent_bookmarks]
    bookmark_recipes = Recipe.query.filter(Recipe.recipe_id.in_(bookmark_recipe_ids)).all()
    from_bookmarks = [recipe.to_dict() for recipe in bookmark_recipes]

    # 2. Recommendations from favorite category (or ranked suggestions)
    from_category = []
    favorite_category = None
    if folder_id:
        subquery = db.session.query(Bookmark.recipe_id).filter_by(user_id=user_id, folder_id=folder_id)
    else:
        subquery = db.session.query(Bookmark.recipe_id).filter_by(user_id=user_id)

    favorite_category_result = db.session.query(
        Recipe.category, func.count(Recipe.category).label('count')
    ).filter(
        Recipe.recipe_id.in_(subquery)
    ).group_by(
        Recipe.category
    ).order_by(
        desc('count')
    ).first()

    if favorite_category_result:
        favorite_category = favorite_category_result[0]
        bookmarked_recipes = db.session.query(Bookmark.recipe_id).filter_by(user_id=user_id)
        category_recipes = Recipe.query.filter(
            Recipe.category == favorite_category,
            ~Recipe.recipe_id.in_(bookmarked_recipes)
        ).order_by(
            desc(Recipe.aggregated_rating)
        ).limit(9).all()
        from_category = [recipe.to_dict() for recipe in category_recipes]

    # 3. Random recommendations (fallback)
    bookmarked_recipes = db.session.query(Bookmark.recipe_id).filter_by(user_id=user_id)
    random_recipes = Recipe.query.filter(
        ~Recipe.recipe_id.in_(bookmarked_recipes)
    ).order_by(
        func.random()
    ).limit(9).all()
    random_suggestions = [recipe.to_dict() for recipe in random_recipes]

    recommendations = [
        {'title': 'From Your Bookmarks', 'recipes': from_bookmarks},
        {'title': 'From a Favorite Category', 'recipes': from_category},
        {'title': 'Random Suggestions', 'recipes': random_suggestions}
    ]
    return jsonify(recommendations), 200


@recommendations_bp.route('/recommendations/refresh', methods=['POST'])
@jwt_required()
@cache.cached(timeout=300)
def refresh_recommendations():
    """
    Force refresh recommendations.
    """
    # Clear the cached recommendations (adjust if using a different caching solution)
    cache.delete_memoized(get_recommendations)
    return get_recommendations()

@recommendations_bp.route("/recommendations/ltr", methods=["GET"])
@jwt_required()
def recommend_for_current_user():
    # If your user is logged in
    user_id = get_jwt_identity()

    # Maybe read `top_k` from query params, or default to 10
    top_k = 10

    # Call your service function
    top_recipes = RecommendService.generate_suggestions_for_user(user_id, top_k=top_k)

    return jsonify({"recommendations": top_recipes})
