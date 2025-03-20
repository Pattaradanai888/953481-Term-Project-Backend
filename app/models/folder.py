from app.extensions import db

class Folder(db.Model):
    __tablename__ = 'folders'
    folder_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    cover_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

    def to_dict(self):
        from app.models.bookmark import Bookmark  # Avoid circular import
        recipe_count = Bookmark.query.filter_by(folder_id=self.folder_id).count()
        return {
            'id': self.folder_id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'coverUrl': self.cover_url,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'recipe_count': recipe_count,
        }