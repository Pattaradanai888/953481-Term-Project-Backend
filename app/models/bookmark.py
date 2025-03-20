from app.extensions import db

class Bookmark(db.Model):
    __tablename__ = 'bookmarks'
    bookmark_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipes.recipe_id'), nullable=False)
    folder_id = db.Column(db.Integer, db.ForeignKey('folders.folder_id'), nullable=False)
    user_rating = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.now())


    def to_dict(self):
        return {
            'bookmark_id': self.bookmark_id,
            'user_id': self.user_id,
            'recipe_id': self.recipe_id,
            'folder_id': self.folder_id,
            'user_rating': self.user_rating,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
        }