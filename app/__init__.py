from flask import Flask
from flask_cors import CORS
from symspellpy import SymSpell, Verbosity
from app.extensions import db, jwt, migrate, cache
import os
from dotenv import load_dotenv

from .api import auth_bp, folder_bp, bookmark_bp, recommendations_bp, bm25_search_bp
from .api import search_bp
from .utils.spell_utils import load_word_frequencies

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
        app.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)  # Reduced to 1 to save memory
        print("10. SymSpell initialized")

        # Add words in batches, limiting total size
        print("11. Adding words to SymSpell in batches")
        batch_size = 5000
        word_count = 0
        max_words = 500000  # Cap total words to avoid memory issues
        batch = {}

        # Sort by frequency (descending) to prioritize common words
        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, freq in sorted_freq:
            if word_count >= max_words:
                break  # Stop after reaching max_words
            if len(word) > 1 and freq >= 2:  # Filter low-frequency and single-letter words
                batch[word] = freq
                word_count += 1

                if len(batch) >= batch_size:
                    for b_word, b_freq in batch.items():
                        app.sym_spell.create_dictionary_entry(b_word, b_freq)
                    print(f"   Added batch of {len(batch)} words. Total: {word_count}")
                    batch = {}

        # Add remaining words in the last batch
        if batch:
            for b_word, b_freq in batch.items():
                app.sym_spell.create_dictionary_entry(b_word, b_freq)
            print(f"   Added final batch of {len(batch)} words. Total: {word_count}")

        # Add specific recipe phrases with high frequency
        print(f"12. Added {word_count} words plus high-priority phrases")

    print("13. Registering blueprints")
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    print("14. Auth blueprint registered")

    app.register_blueprint(search_bp, url_prefix='/api')
    print("15. Search blueprint registered")

    app.register_blueprint(folder_bp, url_prefix='/api/folder')
    print("16. Folder blueprint registered")

    app.register_blueprint(bookmark_bp, url_prefix='/api/bookmark')
    print("17. Bookmark blueprint registered")

    app.register_blueprint(recommendations_bp, url_prefix='/api/')
    print("18. Recommendations blueprint registered")

    app.register_blueprint(bm25_search_bp, url_prefix='/api/')

    app.static_folder = 'static'
    print("19. Static folder set")

    print("20. App creation complete")
    return app