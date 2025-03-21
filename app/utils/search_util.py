from rank_bm25 import BM25Okapi
import numpy as np
from sqlalchemy import func
from fuzzywuzzy import process
from flask import current_app


class RecipeSearchUtil:
    def __init__(self):
        self.bm25 = None
        self.recipes = None
        self.corpus = None
        self.tokenized_corpus = None

    def initialize(self, db_session):
        """Initialize BM25 with all recipes from database."""
        from app.models.recipe import Recipe

        # Get all recipes
        self.recipes = Recipe.query.all()

        # Create tokenized corpus for BM25
        self.tokenized_corpus = []
        self.corpus = []

        for recipe in self.recipes:
            # Create a text representation of the recipe
            ingredients_text = ' '.join([i['name'] for i in recipe.to_dict()['ingredients']])
            text = f"{recipe.name} {recipe.description} {ingredients_text}"
            self.corpus.append(text)
            self.tokenized_corpus.append(text.split())

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        return self

    def correct_query(self, query):
        """Correct typos in query using fuzzywuzzy."""
        if not query:
            return query

        # Use SymSpell from the app context if available
        if hasattr(current_app, 'sym_spell'):
            suggestions = current_app.sym_spell.lookup_compound(query, max_edit_distance=2)
            best_suggestion = suggestions[0] if suggestions else None
            corrected_query = best_suggestion.term if best_suggestion and best_suggestion.distance > 0 else query
            return corrected_query, corrected_query != query

        # Fallback to fuzzywuzzy
        terms = query.split()
        vocabulary = set()
        for doc in self.tokenized_corpus:
            vocabulary.update(doc)

        corrected = [process.extractOne(term, vocabulary)[0] for term in terms]
        corrected_query = " ".join(corrected)

        return corrected_query, corrected_query != query

    def hybrid_search(self, query, page=1, per_page=20, db_session=None):
        """
        Combine PostgreSQL tsvector search with BM25 ranking.
        """
        from app.models.recipe import Recipe

        # Check if BM25 is initialized
        if not self.bm25:
            self.initialize(db_session)

        # Correct query
        search_term, was_corrected = self.correct_query(query)
        corrected_query = search_term if was_corrected else None

        # Step 1: Get candidates using PostgreSQL's full-text search (broader recall)
        search_query = func.plainto_tsquery('english', search_term)
        base_query = Recipe.query.filter(
            Recipe.search_vector.op('@@')(search_query)
        )

        # Get all candidate IDs
        candidate_ids = [recipe.id for recipe in base_query.all()]
        total_results = len(candidate_ids)

        # If no results with PostgreSQL search, try similarity search
        if not candidate_ids and not corrected_query:
            similar = Recipe.query.filter(
                func.similarity(Recipe.name, query) > 0.3
            ).order_by(func.similarity(Recipe.name, query).desc()).first()

            if similar:
                corrected_query = similar.name
                search_term = corrected_query
                search_query = func.plainto_tsquery('english', search_term)
                base_query = Recipe.query.filter(
                    Recipe.search_vector.op('@@')(search_query)
                )
                candidate_ids = [recipe.id for recipe in base_query.all()]
                total_results = len(candidate_ids)

        # Step 2: Re-rank candidates using BM25
        if candidate_ids:
            # Get tokenized query
            tokenized_query = search_term.split()

            # Map database IDs to corpus indices
            id_to_index = {recipe.id: idx for idx, recipe in enumerate(self.recipes)}

            # Filter valid indices (recipes that exist in our BM25 index)
            valid_indices = [id_to_index[recipe_id] for recipe_id in candidate_ids
                             if recipe_id in id_to_index]

            # Get BM25 scores for these indices
            scores = self.bm25.get_scores(tokenized_query)

            # Create pairs of (index, score) for valid indices only
            index_scores = [(idx, scores[idx]) for idx in valid_indices]

            # Sort by score in descending order
            sorted_index_scores = sorted(index_scores, key=lambda x: x[1], reverse=True)

            # Paginate
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_indices = [idx for idx, score in sorted_index_scores[start_idx:end_idx]]

            # Get the corresponding recipes
            results = [self.recipes[idx] for idx in paginated_indices]
        else:
            results = []

        response = {
            'results': [recipe.to_dict() for recipe in results],
            'total': total_results,
            'page': page,
            'per_page': per_page,
            'correctedQuery': corrected_query,
        }

        return response