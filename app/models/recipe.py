from app.extensions import db
from sqlalchemy.dialects.postgresql import TSVECTOR

class Recipe(db.Model):
    __tablename__ = 'recipes'
    __table_args__ = (
        db.Index('idx_search_vector', 'search_vector', postgresql_using='gin'),
    )

    recipe_id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    author_id = db.Column(db.BigInteger, db.ForeignKey('authors.author_id'))
    description = db.Column(db.Text)
    category = db.Column(db.String(255))
    date_published = db.Column(db.DateTime)
    cook_time = db.Column(db.BigInteger)
    prep_time = db.Column(db.BigInteger)
    total_time = db.Column(db.BigInteger)
    aggregated_rating = db.Column(db.Double)
    review_count = db.Column(db.Integer)
    recipe_servings = db.Column(db.Integer)
    recipe_yield = db.Column(db.String(255))
    nutrition_id = db.Column(db.BigInteger, db.ForeignKey('nutrition_info.nutrition_id'))
    search_vector = db.Column(TSVECTOR, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Relationships (optional for detailed views)
    instructions = db.relationship('RecipeInstruction', backref='recipe', lazy='dynamic')

    def to_dict(self):
        primary_image = self.images.filter_by(is_primary=True).first()
        return {
            'recipe_id': self.recipe_id,
            'name': self.name,
            'description': self.description,
            'image_url': primary_image.url if primary_image else None,
        }