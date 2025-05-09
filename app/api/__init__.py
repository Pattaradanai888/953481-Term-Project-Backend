from flask import Blueprint

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__)
search_bp = Blueprint('search', __name__)
folder_bp = Blueprint('folder', __name__)
bookmark_bp = Blueprint('bookmark', __name__)
recommendations_bp = Blueprint('recommendations', __name__)
bm25_search_bp = Blueprint('bm25_search', __name__)

# Import routes to register them with the blueprint
from . import auth
from . import search
from . import folder
from . import bookmark
from . import recommendations
from . import bm25_search
