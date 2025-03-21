from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from flask import current_app
import pickle
import os
from datetime import datetime

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class BM25RecipeSearch:
    def __init__(self, processed_data_path=None, preprocessed_df=None, cache_dir='./cache'):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.bm25 = None
        self.tokenized_corpus = None
        self.recipe_data = None
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Load data if path is provided
        if processed_data_path:
            self.load_data(processed_data_path)
        elif preprocessed_df is not None:
            self.load_from_dataframe(preprocessed_df)

    def load_data(self, processed_data_path):
        """Load preprocessed data from CSV file"""
        print(f"Loading preprocessed data from {processed_data_path}...")

        # Check if we have a cached BM25 index
        cache_file = os.path.join(self.cache_dir, f"bm25_index_{os.path.basename(processed_data_path)}.pkl")

        if os.path.exists(cache_file):
            try:
                print(f"Loading BM25 index from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25 = cache_data['bm25']
                    self.tokenized_corpus = cache_data['tokenized_corpus']
                    self.recipe_data = cache_data['recipe_data']
                print("Successfully loaded BM25 index from cache")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Rebuilding index...")

        # Load data and build index
        df = pd.read_csv(processed_data_path)
        self.load_from_dataframe(df)

        # Cache the index
        try:
            print(f"Caching BM25 index to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'tokenized_corpus': self.tokenized_corpus,
                    'recipe_data': self.recipe_data,
                    'timestamp': datetime.now()
                }, f)
            print("Successfully cached BM25 index")
        except Exception as e:
            print(f"Error caching BM25 index: {e}")

    def load_from_dataframe(self, df):
        """Load preprocessed data from DataFrame"""
        print("Building BM25 index from DataFrame...")

        # Keep the DataFrame for later reference
        self.recipe_data = df

        # Create corpus for BM25
        if 'search_text' in df.columns:
            print("Using preprocessed 'search_text' column")
            corpus = df['search_text'].fillna('').tolist()
        else:
            print("WARNING: 'search_text' column not found. Creating basic search text from available columns.")
            # Fallback to basic text fields
            text_fields = ['Name', 'Description']
            available_fields = [field for field in text_fields if field in df.columns]

            if not available_fields:
                raise ValueError("No searchable text columns found in dataframe")

            corpus = df[available_fields[0]].fillna('').astype(str)
            for field in available_fields[1:]:
                corpus = corpus + " " + df[field].fillna('').astype(str)
            corpus = corpus.tolist()

        # Tokenize corpus
        print("Tokenizing corpus...")
        self.tokenized_corpus = [doc.split() for doc in corpus]

        # Create BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index built successfully")

    def preprocess_query(self, query):
        """Preprocess the search query similar to how we processed the corpus"""
        if not query:
            return []

        # Convert to lowercase
        query = query.lower()

        # Tokenize
        tokens = nltk.word_tokenize(query)

        # Remove stopwords and apply stemming
        processed_tokens = [
            self.stemmer.stem(token) for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

        return processed_tokens

    def search(self, query, page=1, per_page=20, min_score=0.1):
        """
        Search recipes using BM25

        Args:
            query (str): Search query
            page (int): Page number (1-indexed)
            per_page (int): Results per page
            min_score (float): Minimum BM25 score to include in results

        Returns:
            dict: Search results with pagination info
        """
        if not self.bm25:
            return {
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'query': query,
                'error': 'Search index not initialized'
            }

        # Preprocess query
        processed_query = self.preprocess_query(query)
        if not processed_query:
            return {
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'query': query
            }

        # Get BM25 scores
        scores = self.bm25.get_scores(processed_query)

        # Get indices of documents with scores above threshold, sorted by score
        valid_doc_indices = [(i, score) for i, score in enumerate(scores) if score > min_score]
        valid_doc_indices.sort(key=lambda x: x[1], reverse=True)

        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Get paginated results
        paginated_indices = valid_doc_indices[start_idx:end_idx]

        # Prepare results
        results = []
        for idx, score in paginated_indices:
            recipe = self.recipe_data.iloc[idx].to_dict()

            # Add search score
            recipe['search_score'] = float(score)

            # Clean up result by removing processing columns if needed
            result = {k: v for k, v in recipe.items()
                      if not k.startswith('processed_') and k != 'search_text'}

            results.append(result)

        return {
            'results': results,
            'total': len(valid_doc_indices),
            'page': page,
            'per_page': per_page,
            'query': query
        }


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Search recipes using BM25')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed recipes CSV file')
    parser.add_argument('--query', type=str, required=True,
                        help='Search query')

    args = parser.parse_args()

    # Initialize search
    searcher = BM25RecipeSearch(args.data)

    # Perform search
    results = searcher.search(args.query, page=1, per_page=5)

    # Display results
    print(f"Search results for '{args.query}':")
    print(f"Found {results['total']} matching recipes")

    for i, recipe in enumerate(results['results'], 1):
        print(f"\n{i}. {recipe.get('Name', 'Unnamed Recipe')} (Score: {recipe['search_score']:.2f})")
        print(f"   ID: {recipe.get('RecipeId', 'N/A')}")
        print(f"   Author: {recipe.get('AuthorName', 'Unknown')}")
        if 'Description' in recipe and recipe['Description']:
            desc = recipe['Description']
            print(f"   Description: {desc[:100]}..." if len(desc) > 100 else f"   Description: {desc}")