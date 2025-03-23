from app.extensions import db

class NutritionInfo(db.Model):
    __tablename__ = 'nutrition_info'

    nutrition_id = db.Column(db.BigInteger, primary_key=True)
    calories = db.Column(db.Integer, nullable=True)
    fat_content = db.Column(db.Float, nullable=True)  # double precision maps to Float
    saturated_fat_content = db.Column(db.Float, nullable=True)
    cholesterol_content = db.Column(db.Float, nullable=True)
    sodium_content = db.Column(db.Float, nullable=True)
    carbohydrate_content = db.Column(db.Float, nullable=True)
    fiber_content = db.Column(db.Float, nullable=True)
    sugar_content = db.Column(db.Float, nullable=True)
    protein_content = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())

    # One-to-one relationship with Recipe (foreign key is in recipes table)
    recipe = db.relationship('Recipe', backref='nutrition', uselist=False)

    def to_dict(self):
        return {
            'nutrition_id': self.nutrition_id,
            'calories': self.calories,
            'fat_content': self.fat_content,
            'saturated_fat_content': self.saturated_fat_content,
            'cholesterol_content': self.cholesterol_content,
            'sodium_content': self.sodium_content,
            'carbohydrate_content': self.carbohydrate_content,
            'fiber_content': self.fiber_content,
            'sugar_content': self.sugar_content,
            'protein_content': self.protein_content,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }