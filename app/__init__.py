from flask import Flask
from flask_cors import CORS
from symspellpy import SymSpell, Verbosity
from app.extensions import db, jwt, migrate, cache
import os
from dotenv import load_dotenv
from .api import auth_bp, folder_bp, bookmark_bp, recommendations_bp, bm25_search_bp
from .api import search_bp
from .utils.spell_utils import load_word_frequencies
from app.utils.bm25 import BM25RecipeSearch  # Import BM25RecipeSearch

load_dotenv()

def create_app():
    app = Flask(__name__, static_folder='../static')
    print("1. App initialized")

    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY'),
        JWT_TOKEN_LOCATION=["headers"],
        JWT_HEADER_NAME="Authorization",
        JWT_HEADER_TYPE="Bearer",
        CACHE_TYPE='simple',
        CACHE_DEFAULT_TIMEOUT=300,
        PROCESSED_RECIPES_PATH=os.getenv('PROCESSED_RECIPES_PATH', "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/recipes_processed.csv")
    )
    print("2. Config loaded")

    CORS(app, supports_credentials=True, resources={
        r"/api/*": {"origins": "http://localhost:3000", "allow_headers": ["Authorization", "Content-Type"]},
        r"/static/*": {"origins": "http://localhost:3000"}
    })
    print("3. CORS initialized")

    db.init_app(app)
    print("4. DB initialized")

    jwt.init_app(app)
    print("5. JWT initialized")

    migrate.init_app(app, db)
    print("6. Migrations initialized")

    cache.init_app(app)
    print("7. Cache initialized")

    with app.app_context():
        print("8. Entering app context")
        word_freq = load_word_frequencies(force_recompute=False)
        print(f"9. Word frequencies loaded: {len(word_freq)} words")

        # Initialize SymSpell
        app.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        print("10. SymSpell initialized")

        # Add words in batches
        batch_size = 5000
        word_count = 0
        max_words = 500000
        batch = {}
        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, freq in sorted_freq:
            if word_count >= max_words:
                break
            if len(word) > 1 and freq >= 2:
                batch[word] = freq
                word_count += 1
                if len(batch) >= batch_size:
                    for b_word, b_freq in batch.items():
                        app.sym_spell.create_dictionary_entry(b_word, b_freq)
                    print(f"   Added batch of {len(batch)} words. Total: {word_count}")
                    batch = {}

        if batch:
            for b_word, b_freq in batch.items():
                app.sym_spell.create_dictionary_entry(b_word, b_freq)
            print(f"   Added final batch of {len(batch)} words. Total: {word_count}")

        print(f"12. Added {word_count} words plus high-priority phrases")

        # Initialize BM25 search
        processed_data_path = app.config['PROCESSED_RECIPES_PATH']
        if not os.path.exists(processed_data_path):
            app.logger.error(f"Processed recipes file not found: {processed_data_path}")
            app.bm25_searcher = None
        else:
            print("13. Initializing BM25 search...")
            app.bm25_searcher = BM25RecipeSearch(processed_data_path)
            print("14. BM25 search initialized successfully")

    print("15. Registering blueprints")
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(search_bp, url_prefix='/api')
    app.register_blueprint(folder_bp, url_prefix='/api/folder')
    app.register_blueprint(bookmark_bp, url_prefix='/api/bookmark')
    app.register_blueprint(recommendations_bp, url_prefix='/api/')
    app.register_blueprint(bm25_search_bp, url_prefix='/api/')

    app.static_folder = 'static'
    print("16. Static folder set")

    print("17. App creation complete")
    return app