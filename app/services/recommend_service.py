# app/services/recommend_service.py
import logging
import pickle
import os
from collections import Counter
from datetime import datetime

import pandas as pd
from app.extensions import db
from app.models import Review, RecipeKeyword, Keyword
from app.models.recipe import Recipe
from app.models.bookmark import Bookmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendService:



    MODEL_PATH = 'C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/app/utils/ltr_model_keywords.pkl'
    with open(MODEL_PATH, "rb") as f:
        ltr_model = pickle.load(f)

    @staticmethod
    def generate_suggestions_for_user(user_id, top_k=10):
        # 1) Gather candidate recipes not yet bookmarked by this user
        candidate_qs = Recipe.query.limit(500).all()

        data = []
        for r in candidate_qs:
            data.append({
                "user_id": user_id,
                "recipe_id": r.recipe_id,
                "category": r.category or "",
                "aggregated_rating": r.aggregated_rating if r.aggregated_rating else 0.0,
            })
        df_candidates = pd.DataFrame(data)

        if "category" in df_candidates.columns:
            df_candidates = pd.get_dummies(df_candidates, columns=["category"], prefix="cat")

        feature_cols = []
        for col in df_candidates.columns:
            if col.startswith("cat_") or col == "aggregated_rating":
                feature_cols.append(col)

        # Use the class variable ltr_model
        needed_cols = set(RecommendService.ltr_model.feature_name())
        for c in needed_cols:
            if c not in df_candidates.columns:
                df_candidates[c] = 0

        df_candidates = df_candidates[RecommendService.ltr_model.feature_name()]

        preds = RecommendService.ltr_model.predict(df_candidates)

        df_candidates["score"] = preds
        df_candidates.sort_values("score", ascending=False, inplace=True)

        topk_recipe_ids = df_candidates["recipe_id"].head(top_k).tolist()

        return topk_recipe_ids


    @staticmethod
    def generate_folder_suggestions(user_id, folder_id, top_k=10):
        folder_bookmarks = Bookmark.query.filter_by(folder_id=folder_id, user_id=user_id).all()
        if not folder_bookmarks:
            return []

        folder_recipe_ids = [b.recipe_id for b in folder_bookmarks]
        folder_recipes = Recipe.query.filter(Recipe.recipe_id.in_(folder_recipe_ids)).all()
        folder_keywords = [kw.name for r in folder_recipes for kw in r.keywords]
        if not folder_keywords:
            return []

        # Improved keyword selection: exclude generic terms
        generic_keywords = {'Easy', '< 60 Mins', '< 15 Mins', '< 30 Mins', 'Quick'}
        keyword_counts = Counter(kw for kw in folder_keywords if kw not in generic_keywords)
        num_keywords = min(len(keyword_counts), max(3, len(folder_recipes) // 2))
        top_keywords = set([kw for kw, count in keyword_counts.most_common(num_keywords)])
        logger.info(f"Top keywords for folder {folder_id}: {top_keywords}")
        avg_folder_rating = sum(r.aggregated_rating or 0 for r in folder_recipes) / len(folder_recipes)

        # Stricter candidate filtering
        subquery = db.session.query(Bookmark.recipe_id).filter(
            Bookmark.folder_id == folder_id,
            Bookmark.user_id == user_id
        ).subquery()

        candidates = []
        all_candidates = Recipe.query.join(RecipeKeyword).join(Keyword).filter(
            ~Recipe.recipe_id.in_(subquery),
            Keyword.name.in_(top_keywords)
        ).distinct().limit(500).all()

        # Higher overlap threshold: 80% for small, 60% for large
        min_overlap = len(top_keywords) * (0.8 if len(folder_recipes) <= 2 else 0.6)
        for r in all_candidates:
            candidate_keywords = set(kw.name for kw in r.keywords)
            if len(candidate_keywords.intersection(top_keywords)) >= min_overlap:
                candidates.append(r)

        if len(candidates) < 20:
            # Stricter fallback: at least 2 keyword matches
            candidates = [r for r in all_candidates if
                          len(set(kw.name for kw in r.keywords).intersection(top_keywords)) >= 2]
            if len(candidates) < 10:
                return []  # Return empty if too few relevant candidates
        logger.info(f"Number of candidates after filtering: {len(candidates)}")

        # Prepare data
        data_rows = []
        for r in candidates:
            latest_review = Review.query.filter_by(recipe_id=r.recipe_id).order_by(Review.date_submitted.desc()).first()
            days_since = (datetime.now() - latest_review.date_submitted).days if latest_review else 0
            days_since = min(days_since, 3650)
            row = {
                "user_id": user_id,
                "recipe_id": r.recipe_id,
                "category": r.category or "",
                "aggregated_rating": min(r.aggregated_rating or 0.0, 5.0) / 5.0,
                "days_since_review": days_since
            }
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        if "category" in df.columns:
            df = pd.get_dummies(df, columns=["category"], prefix="cat")

        needed_features = RecommendService.ltr_model.feature_name()
        missing_cols = [col for col in needed_features if col not in df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, missing_df], axis=1)
        df = df[needed_features]

        # Predict and boost with stronger similarity
        preds = RecommendService.ltr_model.predict(df)
        similarity_scores = []
        for i, row in df.iterrows():
            candidate_keywords = set(kw.name for kw in candidates[i].keywords)
            overlap = len(candidate_keywords.intersection(top_keywords)) / len(top_keywords)
            rating_diff = abs((row["aggregated_rating"] * 5.0) - avg_folder_rating)
            similarity = overlap * 10.0 - min(rating_diff / 20.0, 0.2)  # Much stronger boost
            similarity_scores.append(similarity)

        # Normalize final scores to positive range
        df = pd.concat([
            df,
            pd.Series(similarity_scores, name="similarity", index=df.index),
            pd.Series(preds + similarity_scores, name="final_score", index=df.index)
        ], axis=1)
        df["final_score"] = (df["final_score"] - df["final_score"].min()) / (
                    df["final_score"].max() - df["final_score"].min() + 1e-6)
        logger.info(f"Top 5 final scores (normalized): {df['final_score'].head(5).tolist()}")

        # Rank and return
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
                "image_url": RecommendService.get_primary_image(recipe_obj),
                "category": recipe_obj.category,
                "aggregated_rating": recipe_obj.aggregated_rating,
                "keywords": [kw.name for kw in recipe_obj.keywords]
            })

        return recommended

    @staticmethod
    def get_primary_image(recipe_obj):
        primary = recipe_obj.images.filter_by(is_primary=True).first()
        return primary.url if primary else None
