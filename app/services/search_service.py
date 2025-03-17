from sqlalchemy import func
from app.extensions import db
from app.models.recipe import Recipe

class SearchService:
    """Service class for handling recipe search logic"""

    @staticmethod
    def search_recipes(query, limit=20):
        """
        Search for recipes based on the provided query using full-text search.
        Returns a list of recipes and an optional corrected query if no results are found.

        Args:
            query (str): The search term provided by the user.
            limit (int): Maximum number of results to return (default is 20).

        Returns:
            tuple: (list of Recipe objects, corrected_query string or None)
        """
        if not query:
            return [], None

        # Convert the query to a tsquery for full-text search
        search_query = func.plainto_tsquery('english', query)

        # Perform the search and rank results by relevance
        results = Recipe.query.filter(
            Recipe.search_vector.op('@@')(search_query)
        ).order_by(
            func.ts_rank(Recipe.search_vector, search_query).desc()
        ).limit(limit).all()

        corrected_query = None
        if not results and ' ' not in query:  # Basic check for potential typos
            # Use pg_trgm similarity to find close matches
            similar = Recipe.query.filter(
                func.similarity(Recipe.name, query) > 0.3
            ).order_by(
                func.similarity(Recipe.name, query).desc()
            ).first()
            if similar:
                corrected_query = similar.name

        return results, corrected_query