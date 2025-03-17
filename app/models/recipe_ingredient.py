from app.extensions import db

class RecipeIngredient(db.Model):
    __tablename__ = 'recipe_ingredients'

    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), primary_key=True)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('ingredients.ingredient_id'), primary_key=True)
    quantity = db.Column(db.String(50))  # e.g., "2", "1.5"
    unit = db.Column(db.String(50))      # e.g., "cups", "tsp"
    preparation_note = db.Column(db.Text)  # e.g., "chopped finely"
    display_order = db.Column(db.Integer, nullable=False, default=0)  # For sorting ingredients in a recipe