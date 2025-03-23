# app/models/keyword.py
from app.extensions import db

class Keyword(db.Model):
    __tablename__ = 'keywords'
    keyword_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

    # Many-to-many relationship with Recipe via RecipeKeyword
    recipes = db.relationship("Recipe", secondary="recipe_keywords", back_populates="keywords")

    def to_dict(self):
        return {"keyword_id": self.keyword_id, "name": self.name}