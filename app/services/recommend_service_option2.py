# app/services/recommend_service_option2.py
import logging
import pickle
import os
from collections import Counter
import pandas as pd
import numpy as np
from app.extensions import db
from app.models import Recipe, RecipeKeyword, Keyword
from app.models.bookmark import Bookmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendServiceOption2:
    MODEL_PATH = 'C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/app/utils/ltr_model_categories_review_count.pkl'
    with open(MODEL_PATH, "rb") as f:
        ltr_model = pickle.load(f)

    @staticmethod
    def generate_folder_suggestions(user_id, folder_id, top_k=10):
        # Fetch bookmarks for the folder and user
        folder_bookmarks = Bookmark.query.filter_by(folder_id=folder_id, user_id=user_id).all()
        if not folder_bookmarks:
            logger.info(f"Folder {folder_id}: No bookmarks found for user {user_id}.")
            return []

        logger.debug(f"Folder {folder_id}: Found {len(folder_bookmarks)} bookmarks.")
        folder_recipe_ids = [b.recipe_id for b in folder_bookmarks]
        folder_recipes = Recipe.query.filter(Recipe.recipe_id.in_(folder_recipe_ids)).all()
        logger.debug(f"Folder {folder_id}: Retrieved {len(folder_recipes)} recipes from bookmarks.")

        folder_categories = [r.category for r in folder_recipes if r.category]
        folder_keywords = [kw.name.lower() for r in folder_recipes for kw in r.keywords]
        folder_names = [r.name.lower() for r in folder_recipes]
        if not folder_categories:
            logger.info(f"Folder {folder_id}: No categories found in recipes.")
            return []

        # Analyze input folder: category and keyword distribution
        category_counts = Counter(folder_categories)
        top_category, top_category_count = category_counts.most_common(1)[0]
        category_ratio = top_category_count / len(folder_recipes)
        logger.info(
            f"Folder {folder_id} Analysis: Dominant category '{top_category}' occurs {top_category_count} times "
            f"({category_ratio:.2f} ratio)."
        )

        # Filter generic keywords and prioritize specific ones
        generic_keywords = {'easy', '< 60 mins', '< 30 mins', '< 15 mins', 'quick', 'for large groups', 'stove top'}
        keyword_counts = Counter(folder_keywords)
        filtered_keywords = {kw: count for kw, count in keyword_counts.items() if kw not in generic_keywords}

        # If insufficient keywords, extract from recipe names
        if len(filtered_keywords) < 3:
            name_words = [word for name in folder_names for word in name.split() if len(word) > 3]
            name_counts = Counter(name_words)
            filtered_keywords.update({kw: count for kw, count in name_counts.items() if kw not in generic_keywords})

        top_keywords = set([kw for kw, _ in Counter(filtered_keywords).most_common(3)])
        if category_ratio < 0.5 or not top_keywords:
            logger.info(
                f"Folder {folder_id}: Top category {top_category} ({category_ratio:.2f}); Keywords extracted: {top_keywords}"
            )
        else:
            top_keywords = {top_category.lower()}
            logger.info(f"Folder {folder_id}: Dominant category '{top_category}' used as sole keyword.")

        avg_folder_rating = sum(r.aggregated_rating or 0 for r in folder_recipes) / len(folder_recipes)
        logger.debug(f"Folder {folder_id}: Average rating of recipes: {avg_folder_rating:.2f}")

        # Candidate filtering
        subquery = db.session.query(Bookmark.recipe_id).filter(
            Bookmark.folder_id == folder_id,
            Bookmark.user_id == user_id
        ).subquery()

        # Try keyword + category filter first
        candidates = Recipe.query.join(RecipeKeyword).join(Keyword).filter(
            ~Recipe.recipe_id.in_(subquery),
            Recipe.category == top_category,
            Keyword.name.in_([kw.capitalize() for kw in top_keywords])  # Match case in DB
        ).distinct().limit(500).all()

        if len(candidates) < 50:  # Softer threshold
            candidates = Recipe.query.filter(
                ~Recipe.recipe_id.in_(subquery),
                Recipe.category == top_category
            ).limit(500).all()
            logger.info(f"Folder {folder_id}: Fallback to category-only filter; candidates: {len(candidates)}")
        if len(candidates) < 10:
            logger.info(f"Folder {folder_id}: Too few candidates ({len(candidates)}); returning empty list.")
            return []
        logger.info(f"Folder {folder_id}: Number of candidates after filtering: {len(candidates)}")

        # Prepare data for LTR model
        data_rows = []
        for r in candidates:
            row = {
                "user_id": user_id,
                "recipe_id": r.recipe_id,
                "category": r.category or "",
                "aggregated_rating": min(r.aggregated_rating or 0.0, 5.0) / 5.0,
                "review_count": r.review_count if r.review_count is not None else 0
            }
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df = pd.get_dummies(df, columns=["category"], prefix="cat")

        needed_features = RecommendServiceOption2.ltr_model.feature_name()
        missing_cols = [col for col in needed_features if col not in df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, missing_df], axis=1)
        df = df[needed_features]

        # Predict scores using LTR model and apply keyword overlap boost
        preds = RecommendServiceOption2.ltr_model.predict(df)
        similarity_scores = []
        for r in candidates:
            candidate_keywords = set(kw.name.lower() for kw in r.keywords)
            candidate_name = r.name.lower()
            overlap = len(candidate_keywords.intersection(top_keywords)) / max(len(top_keywords), 1)
            name_bonus = 1.0 if any(kw in candidate_name for kw in top_keywords) else 0.0
            similarity_scores.append((overlap + name_bonus) * 1.5)  # Stronger boost with name match

        final_scores = preds + np.array(similarity_scores)
        if np.std(final_scores) < 1e-6:
            final_scores = similarity_scores
        df["final_score"] = (final_scores - final_scores.min()) / (
            final_scores.max() - final_scores.min() + 1e-6)
        logger.info(f"Folder {folder_id}: Top 5 normalized final scores: {df['final_score'].head(5).tolist()}")

        # Rank candidates based on final scores
        df_with_scores = pd.DataFrame({"recipe_index": range(len(candidates)), "score": df["final_score"]})
        df_with_scores.sort_values("score", ascending=False, inplace=True)
        topk_indices = df_with_scores.head(top_k)["recipe_index"].tolist()

        recommended = []
        for idx in topk_indices:
            recipe_obj = candidates[idx]
            recommended.append({
                "recipe_id": recipe_obj.recipe_id,
                "name": recipe_obj.name,
                "description": recipe_obj.description,
                "image_url": RecommendServiceOption2.get_primary_image(recipe_obj),
                "category": recipe_obj.category,
                "aggregated_rating": recipe_obj.aggregated_rating,
                "keywords": [kw.name for kw in recipe_obj.keywords]
            })

        # Debug: Compare input folder dominant category with recommended recipes
        match_count = sum(1 for rec in recommended if rec.get("category") == top_category)
        logger.info(
            f"Folder {folder_id}: {match_count} out of {len(recommended)} recommended recipes match the dominant category '{top_category}'."
        )

        return recommended

    @staticmethod
    def get_primary_image(recipe_obj):
        primary = recipe_obj.images.filter_by(is_primary=True).first()
        return primary.url if primary else None
