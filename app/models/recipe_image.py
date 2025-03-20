from app.extensions import db

class RecipeImage(db.Model):
    __tablename__ = 'recipe_images'

    image_id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    is_primary = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())