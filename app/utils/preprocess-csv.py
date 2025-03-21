import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import json

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


class RecipePreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Add cooking-specific stop words
        self.stop_words.update(['tsp', 'tbsp', 'cup', 'cups', 'oz', 'lb', 'g', 'kg', 'ml', 'l'])

    def preprocess_text(self, text):
        """
        Preprocess text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Removing numbers
        4. Removing stop words
        5. Stemming
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and apply stemming
        processed_tokens = [
            self.stemmer.stem(token) for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

        return " ".join(processed_tokens)

    def preprocess_ingredients(self, ingredients_list):
        """
        Process a list of ingredients, extracting just the food items
        and removing measurements, etc.
        """
        if not ingredients_list:
            return ""

        # Join all ingredients
        if isinstance(ingredients_list, list):
            if all(isinstance(item, dict) and 'text' in item for item in ingredients_list):
                # Handle case where ingredients are dictionaries with 'text' key
                ingredients_text = " ".join([item['text'] for item in ingredients_list])
            elif all(isinstance(item, str) for item in ingredients_list):
                # Handle case where ingredients are plain strings
                ingredients_text = " ".join(ingredients_list)
            else:
                # Try to convert to string as a fallback
                ingredients_text = " ".join([str(item) for item in ingredients_list])
        elif isinstance(ingredients_list, str):
            # If it's already a string
            ingredients_text = ingredients_list
        else:
            # Try to convert to string as a last resort
            try:
                ingredients_text = str(ingredients_list)
            except:
                return ""

        # Remove measurements and common preparation instructions
        # This is a simplified approach - a more comprehensive solution might use NER
        measurement_patterns = [
            r'\d+\s*(?:cup|cups|tablespoon|tbsp|teaspoon|tsp|ounce|oz|pound|lb|gram|g|kg|ml|l)',
            r'(?:chopped|minced|diced|sliced|grated|crushed|ground)',
            r'(?:to taste|optional|as needed)'
        ]

        for pattern in measurement_patterns:
            ingredients_text = re.sub(pattern, '', ingredients_text, flags=re.IGNORECASE)

        return self.preprocess_text(ingredients_text)

    def preprocess_recipe_data(self, recipes_df):
        """
        Preprocess the entire recipes dataframe
        """
        preprocessor = self

        # Create a copy to avoid modifying the original
        df = recipes_df.copy()

        # Preprocess text fields
        print("Preprocessing recipe names...")
        df['processed_name'] = df['Name'].apply(preprocessor.preprocess_text)

        print("Preprocessing descriptions...")
        df['processed_description'] = df['Description'].apply(preprocessor.preprocess_text)

        # Handle ingredients (which might be in a different format depending on your data)
        print("Preprocessing ingredients...")

        # If ingredients are in a separate column as JSON strings
        if 'RecipeIngredientParts' in df.columns:
            # Parse JSON strings if needed
            try:
                df['ingredients_list'] = df['RecipeIngredientParts'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
                df['processed_ingredients'] = df['ingredients_list'].apply(preprocessor.preprocess_ingredients)
            except:
                print("Error processing RecipeIngredientParts column - check format")
                df['processed_ingredients'] = ""

        # If ingredients are already parsed in your dataframe, adjust accordingly

        # Combine all processed text for search
        print("Creating combined search text...")
        df['search_text'] = (
                df['processed_name'] + " " +
                df['processed_description'] + " " +
                df.get('processed_ingredients', '')
        ).str.strip()

        return df


def create_preprocessed_dataset(recipes_path):
    """
    Load and preprocess the recipes dataset
    """
    print(f"Loading recipes from {recipes_path}...")
    recipes_df = pd.read_csv(recipes_path)

    print(f"Loaded {len(recipes_df)} recipes")

    # Check columns
    print("Available columns:")
    for col in recipes_df.columns:
        print(f"- {col}")

    # Initialize preprocessor
    preprocessor = RecipePreprocessor()

    # Preprocess data
    processed_df = preprocessor.preprocess_recipe_data(recipes_df)

    # Save processed data
    output_path = recipes_path.replace('.csv', '_processed.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to {output_path}")

    return processed_df


if __name__ == "__main__":
    # Replace with your actual path
    recipes_path = "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/recipes.csv"
    processed_data = create_preprocessed_dataset(recipes_path)

    # Display sample of processed data
    print("\nSample of processed data:")
    print(processed_data[['Name', 'processed_name', 'search_text']].head())