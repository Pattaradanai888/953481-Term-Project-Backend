from app import db


class RecipeKeyword(db.Model):
    __tablename__ = 'recipe_keywords'
    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), primary_key=True)
    keyword_id = db.Column(db.Integer, db.ForeignKey('keywords.keyword_id'), primary_key=True)

    def to_dict(self):
        return {"recipe_id": self.recipe_id, "keyword_id": self.keyword_id}