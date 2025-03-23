# app/models/recommendation.py
from app.extensions import db

class Recommendation(db.Model):
    __tablename__ = 'recommendations'

    recommendation_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), nullable=False)
    folder_id = db.Column(db.Integer, db.ForeignKey('folders.folder_id'), nullable=True)
    score = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    is_displayed = db.Column(db.Boolean, nullable=True)
    displayed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'recommendation_id': self.recommendation_id,
            'user_id': self.user_id,
            'recipe_id': self.recipe_id,
            'folder_id': self.folder_id,
            'score': self.score,
            'type': self.type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_displayed': self.is_displayed,
            'displayed_at': self.displayed_at.isoformat() if self.displayed_at else None
        }