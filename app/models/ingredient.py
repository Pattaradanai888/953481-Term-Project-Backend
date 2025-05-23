from app.extensions import db

class Ingredient(db.Model):
    __tablename__ = 'ingredients'

    ingredient_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)