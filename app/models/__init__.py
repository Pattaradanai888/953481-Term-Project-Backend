# app/models/__init__.py
from .author import Author
from .bookmark import Bookmark
from .category import Category
from .folder import Folder
from .ingredient import Ingredient
from .keyword import Keyword
from .nutrition_info import NutritionInfo
from .recipe import Recipe
from .recipe_category import RecipeCategory
from .recipe_image import RecipeImage
from .recipe_ingredient import RecipeIngredient
from .recipe_instruction import RecipeInstruction
from .recipe_keyword import RecipeKeyword
from .review import Review
from .user import User
from .recommendation import Recommendation
from .search_history import SearchHistory

__all__ = [
    'Author', 'Bookmark', 'Category', 'Folder', 'Ingredient', 'Keyword',
    'NutritionInfo', 'Recipe', 'RecipeCategory', 'RecipeImage', 'RecipeIngredient',
    'RecipeInstruction', 'RecipeKeyword', 'Review', 'Recommendation',
    'User' , 'SearchHistory'
]