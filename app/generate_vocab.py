# generate_vocab.py
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import create_app  # Import your Flask app factory
from app.utils.spell_utils import load_word_frequencies

def main():
    # Create Flask app to initialize extensions (db, etc.)
    app = create_app()

    # Run within app context to access database
    with app.app_context():
        # Force recompute to generate fresh vocabulary
        word_freq = load_word_frequencies(force_recompute=True)

    print("Vocabulary generation completed!")

if __name__ == "__main__":
    main()