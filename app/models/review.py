# app/models/review.py
from datetime import datetime
from app.extensions import db

class Review(db.Model):
    __tablename__ = 'reviews'

    review_id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), nullable=False)
    author_id = db.Column(db.BigInteger, db.ForeignKey('authors.author_id'), nullable=True)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 or 1-10
    review_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    date_submitted = db.Column(db.DateTime,nullable=False)
    date_modified = db.Column(db.DateTime, nullable=False)

    # Relationships to other models if desired:
    # "User" and "Recipe" are other models defined in your code

    def to_dict(self):
        """Helper method to serialize Review into a dict."""
        return {
            "review_id": self.review_id,
            "author_id": self.author_id,
            "recipe_id": self.recipe_id,
            "rating": self.rating,
            "review_text": self.review_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "date_submitted": self.date_submitted,
            "date_modified": self.date_modified
        }