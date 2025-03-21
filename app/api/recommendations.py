# app/api/recommendations.py
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.models import Recipe, Bookmark
from sqlalchemy import func, desc
from app.extensions import db
from . import recommendations_bp

@recommendations_bp.route('/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    user_id = get_jwt_identity()
    folder_id = request.args.get('folder_id', type=int)

    # 1. Recommendations from user's bookmarks
    bookmark_query = Bookmark.query.filter_by(user_id=user_id)

    if folder_id:
        bookmark_query = bookmark_query.filter_by(folder_id=folder_id)

    # Get recent bookmarks
    recent_bookmarks = bookmark_query.order_by(desc(Bookmark.created_at)).limit(3).all()
    bookmark_recipe_ids = [bookmark.recipe_id for bookmark in recent_bookmarks]
    bookmark_recipes = Recipe.query.filter(Recipe.recipe_id.in_(bookmark_recipe_ids)).all()

    # Format bookmark recipes
    from_bookmarks = []
    for recipe in bookmark_recipes:
        recipe_dict = recipe.to_dict()
        from_bookmarks.append(recipe_dict)

    # 2. Recommendations from favorite category
    favorite_category = None
    from_category = []

    # Find favorite category based on user's bookmarks
    if folder_id:
        # Get favorite category from specific folder
        subquery = db.session.query(
            Bookmark.recipe_id
        ).filter_by(user_id=user_id, folder_id=folder_id)
    else:
        # Get favorite category from all bookmarks
        subquery = db.session.query(
            Bookmark.recipe_id
        ).filter_by(user_id=user_id)

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

        # Get recipes from favorite category (excluding already bookmarked ones)
        bookmarked_recipes = db.session.query(Bookmark.recipe_id).filter_by(user_id=user_id)
        category_recipes = Recipe.query.filter(
            Recipe.category == favorite_category,
            ~Recipe.recipe_id.in_(bookmarked_recipes)
        ).order_by(
            desc(Recipe.aggregated_rating)
        ).limit(3).all()

        # Format category recipes
        for recipe in category_recipes:
            recipe_dict = recipe.to_dict()
            from_category.append(recipe_dict)

    # 3. Random recommendations
    # Get random recipes (excluding already bookmarked ones)
    bookmarked_recipes = db.session.query(Bookmark.recipe_id).filter_by(user_id=user_id)
    random_recipes = Recipe.query.filter(
        ~Recipe.recipe_id.in_(bookmarked_recipes)
    ).order_by(
        func.random()
    ).limit(3).all()

    # Format random recipes
    random_suggestions = []
    for recipe in random_recipes:
        recipe_dict = recipe.to_dict()
        random_suggestions.append(recipe_dict)

    # Combine all recommendations
    recommendations = [
        {'title': 'From Your Bookmarks', 'recipes': from_bookmarks},
        {'title': 'From a Favorite Category', 'recipes': from_category},
        {'title': 'Random Suggestions', 'recipes': random_suggestions}
    ]

    return jsonify(recommendations), 200


@recommendations_bp.route('/recommendations/refresh', methods=['POST'])
@jwt_required()
def refresh_recommendations():
    """
    Force refresh recommendations for a user
    (This is an optional endpoint that could be used if you want to
    implement a "Refresh" button separate from the main recommendations endpoint)
    """
    user_id = get_jwt_identity()
    folder_id = request.json.get('folder_id')

    # For now, this just calls the same logic as the GET endpoint
    # but you could add additional logic here if needed

    return get_recommendations()