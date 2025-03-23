from app.extensions import db

class Author(db.Model):
    __tablename__ = 'authors'

    author_id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())

    # One-to-many relationships
    recipes = db.relationship('Recipe', backref='author', lazy='dynamic')
    reviews = db.relationship('Review', backref='author', lazy='dynamic')

    def to_dict(self):
        return {
            'author_id': self.author_id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }