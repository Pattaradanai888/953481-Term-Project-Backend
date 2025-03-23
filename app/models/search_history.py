# app/models/search_history.py
from app.extensions import db

class SearchHistory(db.Model):
    __tablename__ = 'search_history'

    history_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'history_id': self.history_id,
            'user_id': self.user_id,
            'query': self.query,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }