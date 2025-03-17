from app.extensions import db

class RecipeInstruction(db.Model):
    __tablename__ = 'recipe_instructions'

    instruction_id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.BigInteger, db.ForeignKey('recipes.recipe_id'), nullable=False)
    step_number = db.Column(db.Integer, nullable=False)
    instruction_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())