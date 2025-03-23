from app.extensions import db

class Category(db.Model):
    __tablename__ = 'categories'

    category_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)

    # Many-to-many relationship with Recipe via recipe_categories
    recipes = db.relationship('Recipe', secondary='recipe_categories', back_populates='categories')

    def to_dict(self):
        return {
            'category_id': self.category_id,
            'name': self.name
        }