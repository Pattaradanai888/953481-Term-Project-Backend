import os
import pickle
from collections import Counter
from tqdm import tqdm

from app.models.recipe import Recipe
from app.models.recipe_instruction import RecipeInstruction  # Assumed model for recipe instructions
from app.models import Ingredient, RecipeIngredient
from app.extensions import db

# Adjust the path for your environment
VOCAB_FILE = 'C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/vocab_cache.pkl'


def load_word_frequencies(force_recompute=False):
    # Load cached vocabulary if available and recomputation is not forced.
    if os.path.exists(VOCAB_FILE) and not force_recompute:
        print(f"Loading cached vocabulary from {VOCAB_FILE}")
        with open(VOCAB_FILE, 'rb') as f:
            return pickle.load(f)

    print("Generating new vocabulary...")
    vocabulary_words = []
    batch_size = 5000
    total_recipes = db.session.query(Recipe).count()
    print(f"Total recipes to process: {total_recipes}")

    # Process recipe names and their cooking instructions in batches.
    for offset in tqdm(range(0, total_recipes, batch_size), desc="Processing recipes"):
        # Query recipe id and name
        recipes = db.session.query(Recipe.recipe_id, Recipe.name)\
            .limit(batch_size).offset(offset).all()
        for recipe_id, name in recipes:
            # Add the recipe name (food name)
            vocabulary_words.append(name.lower())

            # Query cooking instructions for this recipe
            instructions = db.session.query(RecipeInstruction.instruction_id) \
                .filter(RecipeInstruction.recipe_id == recipe_id).all()
            # Each instruction is a tuple (instruction,), so process each one.
            for (instr,) in instructions:
                # Convert instr to a string in case it isn't one already.
                vocabulary_words.extend(str(instr).lower().split())

    # Fetch unique ingredient names
    print("Fetching ingredient names...")
    ingredient_query = (
        db.session.query(Ingredient.name)
        .join(RecipeIngredient, Ingredient.ingredient_id == RecipeIngredient.ingredient_id)
        .distinct()
    ).all()
    ingredient_names = [i.name.lower() for i in tqdm(ingredient_query, desc="Processing ingredients")]

    # Combine words from recipe names, cooking instructions, ingredient names, and the predefined keywords.
    all_words = vocabulary_words + ingredient_names

    # Count the frequencies of each word
    print("Combining and counting word frequencies...")
    words = " ".join(all_words).split()
    word_freq = Counter(words)
    print(f"Total unique words: {len(word_freq)}")

    # Ensure the cache directory exists and save the vocabulary
    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(word_freq, f)
    print(f"Vocabulary saved to {VOCAB_FILE}")

    return word_freq
