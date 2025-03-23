# app/models/recipe_category.py
from app.extensions import db

class RecipeCategory(db.Model):
    __tablename__ = 'recipe_categories'

    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.category_id'), primary_key=True)