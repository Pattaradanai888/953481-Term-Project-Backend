import csv
from io import StringIO

import pandas as pd
import re
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
import isodate
import json
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from functools import partial
import time

# Configuration
DB_CONFIG = {
    'dbname': 'ir-project',
    'user': 'postgres',
    'password': 'plzmedota2',  # Change this to your password
    'host': 'localhost',
    'port': '5432'
}


def parse_r_list(text):
    """Enhanced parser for R-style lists, JSON arrays, and various string formats"""
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()
    original_text = text  # Keep for error reporting

    # Regular expression to detect URLs
    url_pattern = re.compile(r'^https?://', re.IGNORECASE)

    try:
        # Case 1: R-style list c("item1", "item2")
        if text.startswith('c('):
            content = text[2:-1].strip()  # Remove c() wrapper
            return list(csv.reader(StringIO(content),
                                   quotechar='"',
                                   skipinitialspace=True,
                                   escapechar='\\'))[0]

        # Case 2: JSON array format
        if text.startswith('[') and text.endswith(']'):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Handle malformed JSON with quoted elements
                return [item.strip().strip('"')
                        for item in text[1:-1].split(',')
                        if item.strip()]

        # Case 3: Newline-separated instructions
        if '\n' in text:
            return [line.strip().strip('"')
                    for line in text.split('\n')
                    if line.strip()]

        # Case 4: Check for URL pattern
        if url_pattern.match(text):
            return [text]

        # Case 5: Comma-separated values without quotes
        if ',' in text:
            return [item.strip().strip('"')
                    for item in text.split(',')
                    if item.strip()]

        # Case 5: Single quoted string
        if (text.startswith('"') and text.endswith('"')) or \
                (text.startswith("'") and text.endswith("'")):
            return [text[1:-1]]

    except Exception as e:
        print(f"Error parsing '{original_text}': {str(e)}")
        return []

    # Final fallback: single unquoted string
    return [text.strip('"')]


def clean_text(text):
    """Clean text without stopword removal or stemming"""
    if not isinstance(text, str):
        return ""

    # Basic cleaning only
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_duration(duration_str):
    """Parse ISO duration strings to seconds"""
    if not isinstance(duration_str, str) or not duration_str.strip():
        return None

    try:
        return int(isodate.parse_duration(duration_str).total_seconds())
    except:
        return None


def process_row(row, columns_to_process):
    """Process a single row with the appropriate function for each column"""
    processed_row = row.copy()

    for col, process_func in columns_to_process.items():
        if col in processed_row:
            processed_row[col] = process_func(processed_row[col])

    return processed_row


def process_chunk(chunk, process_func):
    return chunk.apply(process_func, axis=1)


def process_dataframe_parallel(df, columns_to_process, num_workers=4):
    process_func = partial(process_row, columns_to_process=columns_to_process)
    chunks = np.array_split(df, num_workers)

    # Prepare arguments for starmap
    args = [(chunk, process_func) for chunk in chunks]

    with mp.Pool(num_workers) as pool:
        processed_chunks = pool.starmap(process_chunk, args)

    return pd.concat(processed_chunks)


def connect_to_db():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to database successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise


def create_test_user(conn):
    """Create a test user for demonstration purposes"""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO users (username, email, password_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO NOTHING
                RETURNING user_id
                """,
                ('testuser', 'test@example.com', 'hashed_password_here')
            )
            user_id = cur.fetchone()
            if user_id:
                print(f"Created test user with ID: {user_id[0]}")
                return user_id[0]
            else:
                # If user already exists, get their ID
                cur.execute("SELECT user_id FROM users WHERE username = 'testuser'")
                user_id = cur.fetchone()[0]
                print(f"Using existing test user with ID: {user_id}")
                return user_id
        except Exception as e:
            print(f"Error creating test user: {e}")
            conn.rollback()
            raise


def process_recipes_data(recipes_df):
    """Process recipes data for normalized schema"""
    print("Processing recipes data...")

    # Define processing rules
    columns_to_process = {
        # Array parsing
        'Keywords': parse_r_list,
        'RecipeIngredientParts': parse_r_list,
        'RecipeIngredientQuantities': parse_r_list,
        'Images': parse_r_list,
        'RecipeInstructions': parse_r_list,

        # Text cleaning
        'RecipeCategory': clean_text,
        'Name': clean_text,
        'Description': clean_text,
        'AuthorName': clean_text,

        # Time conversion
        'CookTime': parse_duration,
        'PrepTime': parse_duration,
        'TotalTime': parse_duration
    }

    processed_df = process_dataframe_parallel(recipes_df, columns_to_process)

    # Map to new schema columns for the main recipe table
    column_mapping = {
        'RecipeId': 'recipe_id',
        'Name': 'name',
        'AuthorId': 'author_id',
        'AuthorName': 'author_name',
        'CookTime': 'cook_time',
        'PrepTime': 'prep_time',
        'TotalTime': 'total_time',
        'DatePublished': 'date_published',
        'Description': 'description',
        'AggregatedRating': 'aggregated_rating',
        'ReviewCount': 'review_count',
        'RecipeServings': 'recipe_servings',
        'RecipeYield': 'recipe_yield',
        'RecipeCategory': 'recipe_category',
    }

    processed_df = processed_df.rename(columns=column_mapping)

    # Add updated_at field
    processed_df['updated_at'] = datetime.now()

    return processed_df


def process_reviews_data(reviews_df):
    """Process and clean the reviews dataframe"""
    print("Processing reviews data...")

    # Define text columns that need cleaning
    text_columns = ['Review', 'AuthorName']

    # Create a dictionary of columns and their processing functions
    columns_to_process = {}

    for col in text_columns:
        if col in reviews_df.columns:
            columns_to_process[col] = clean_text

    # Process the dataframe in parallel
    processed_df = process_dataframe_parallel(reviews_df, columns_to_process)

    # Rename columns to match database schema
    column_mapping = {
        'ReviewId': 'review_id',
        'RecipeId': 'recipe_id',
        'AuthorId': 'author_id',
        'AuthorName': 'author_name',
        'Rating': 'rating',
        'Review': 'review_text',
        'DateSubmitted': 'date_submitted',
        'DateModified': 'date_modified'
    }

    # Apply the column mapping
    processed_df = processed_df.rename(
        columns={old: new for old, new in column_mapping.items() if old in processed_df.columns})

    # Ensure all expected columns exist
    for col in column_mapping.values():
        if col not in processed_df.columns:
            processed_df[col] = None

    return processed_df


def insert_nutrition_info(conn, recipes_df):
    """Insert nutrition information and return a mapping of recipe_id to nutrition_id"""
    print("Inserting nutrition information...")
    nutrition_mapping = {}

    # Extract nutrition columns
    nutrition_columns = [
        'calories', 'fat_content', 'saturated_fat_content',
        'cholesterol_content', 'sodium_content', 'carbohydrate_content',
        'fiber_content', 'sugar_content', 'protein_content'
    ]

    # Prepare nutrition data
    nutrition_data = []
    for idx, row in recipes_df.iterrows():
        nutrition_values = [row.get(col) for col in nutrition_columns]
        # Only insert if we have at least one non-None value
        if any(v is not None for v in nutrition_values):
            nutrition_data.append(nutrition_values)

    if not nutrition_data:
        return nutrition_mapping

    with conn.cursor() as cur:
        try:
            # Insert nutrition data
            query = f"""
                INSERT INTO nutrition_info (
                    calories, fat_content, saturated_fat_content, 
                    cholesterol_content, sodium_content, carbohydrate_content,
                    fiber_content, sugar_content, protein_content
                )
                VALUES ({', '.join(['%s'] * len(nutrition_columns))})
                RETURNING nutrition_id
            """

            # Execute in batches
            batch_size = 1000
            for i in range(0, len(nutrition_data), batch_size):
                batch = nutrition_data[i:i + batch_size]

                for values in batch:
                    cur.execute(query, values)
                    nutrition_id = cur.fetchone()[0]
                    # Map to the recipe_id
                    recipe_idx = i + batch.index(values)
                    recipe_id = recipes_df.iloc[recipe_idx]['recipe_id']
                    nutrition_mapping[recipe_id] = nutrition_id

                conn.commit()

            print(f"Inserted {len(nutrition_mapping)} nutrition records")
            return nutrition_mapping

        except Exception as e:
            print(f"Error inserting nutrition info: {e}")
            conn.rollback()
            return {}


def insert_authors(conn, recipes_df):
    """Insert authors and return a mapping of author_id to database id"""
    print("Inserting authors...")

    # Extract unique authors
    unique_authors = recipes_df[['author_id', 'author_name']].drop_duplicates()
    unique_authors = unique_authors[~unique_authors['author_id'].isna()]

    if unique_authors.empty:
        return {}

    with conn.cursor() as cur:
        author_count = 0
        for _, row in unique_authors.iterrows():
            try:
                cur.execute(
                    """
                    INSERT INTO authors (author_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (author_id) DO UPDATE
                    SET name = EXCLUDED.name
                    """,
                    (row['author_id'], row['author_name'])
                )
                author_count += 1
            except Exception as e:
                print(f"Error inserting author {row['author_name']}: {e}")
                conn.rollback()

        conn.commit()
        print(f"Inserted/updated {author_count} authors")

    return True


def insert_keywords(conn, recipes_df):
    """Insert keywords and return a mapping of keyword text to keyword_id"""
    print("Inserting keywords...")

    # Extract all unique keywords from recipes
    all_keywords = set()
    for _, row in recipes_df.iterrows():
        if 'Keywords' in row and isinstance(row['Keywords'], list):
            all_keywords.update(row['Keywords'])

    keyword_mapping = {}
    if not all_keywords:
        return keyword_mapping

    with conn.cursor() as cur:
        for keyword in all_keywords:
            try:
                cur.execute(
                    """
                    INSERT INTO keywords (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING keyword_id
                    """,
                    (keyword,)
                )
                result = cur.fetchone()
                if result:
                    keyword_mapping[keyword] = result[0]
                else:
                    # If keyword already exists, get its ID
                    cur.execute("SELECT keyword_id FROM keywords WHERE name = %s", (keyword,))
                    keyword_mapping[keyword] = cur.fetchone()[0]
            except Exception as e:
                print(f"Error inserting keyword '{keyword}': {e}")
                conn.rollback()

        conn.commit()
        print(f"Inserted {len(keyword_mapping)} unique keywords")

    return keyword_mapping


def insert_ingredients(conn, recipes_df):
    """Insert ingredients and return a mapping of ingredient text to ingredient_id"""
    print("Inserting ingredients...")

    # Extract all unique ingredients from recipes
    all_ingredients = set()
    for _, row in recipes_df.iterrows():
        if 'RecipeIngredientParts' in row and isinstance(row['RecipeIngredientParts'], list):
            all_ingredients.update(row['RecipeIngredientParts'])

    ingredient_mapping = {}
    if not all_ingredients:
        return ingredient_mapping

    with conn.cursor() as cur:
        for ingredient in all_ingredients:
            try:
                cur.execute(
                    """
                    INSERT INTO ingredients (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING ingredient_id
                    """,
                    (ingredient,)
                )
                result = cur.fetchone()
                if result:
                    ingredient_mapping[ingredient] = result[0]
                else:
                    # If ingredient already exists, get its ID
                    cur.execute("SELECT ingredient_id FROM ingredients WHERE name = %s", (ingredient,))
                    ingredient_mapping[ingredient] = cur.fetchone()[0]
            except Exception as e:
                print(f"Error inserting ingredient '{ingredient}': {e}")
                conn.rollback()

        conn.commit()
        print(f"Inserted {len(ingredient_mapping)} unique ingredients")

    return ingredient_mapping


def insert_recipes(conn, recipes_df, nutrition_mapping):
    """Insert processed recipes into the database"""
    print("Inserting recipes into database...")

    # Create a cursor for bulk insertion
    with conn.cursor() as cur:
        # Get the list of columns in the target table
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'recipes'")
        db_columns = [row[0] for row in cur.fetchall()]

        if 'created_at' in db_columns:
            db_columns.remove('created_at')

        # Add search_vector to columns to exclude
        if 'search_vector' in db_columns:
            db_columns.remove('search_vector')

        # Prepare recipe data for insertion
        inserted_count = 0

        # Use a batch size to avoid memory issues
        batch_size = 1000

        # Convert DataFrame column types to match PostgreSQL expectations
        recipes_df['aggregated_rating'] = pd.to_numeric(recipes_df['aggregated_rating'], errors='coerce')
        recipes_df['review_count'] = pd.to_numeric(recipes_df['review_count'], errors='coerce')
        recipes_df['recipe_servings'] = pd.to_numeric(recipes_df['recipe_servings'], errors='coerce')

        # Convert date columns to proper datetime format
        try:
            recipes_df['date_published'] = pd.to_datetime(recipes_df['date_published'], errors='coerce')
        except:
            recipes_df['date_published'] = None

        # Add nutrition_id from mapping
        recipes_df['nutrition_id'] = recipes_df['recipe_id'].map(nutrition_mapping)

        # Process in batches
        with tqdm(total=len(recipes_df)) as pbar:
            for i in range(0, len(recipes_df), batch_size):
                batch = recipes_df.iloc[i:i + batch_size]

                # Prepare data for this batch
                values_list = []
                for _, row in batch.iterrows():
                    # Create a dictionary of values for this row
                    row_dict = {}
                    for col in db_columns:
                        if col in row:
                            value = row[col]
                            # Handle NaN, None, and special data types
                            if np.isscalar(value) and pd.isna(value):
                                row_dict[col] = None
                            else:
                                row_dict[col] = value
                        else:
                            row_dict[col] = None

                    values_list.append(row_dict)

                # Skip if no valid rows in this batch
                if not values_list:
                    continue

                try:
                    # Use execute_values for efficient batch insertion
                    columns = [col for col in db_columns if col in values_list[0]]
                    values_template = "(" + ", ".join(["%s"] * len(columns)) + ")"

                    # Prepare the values in the order of columns
                    values = [[row_dict.get(col) for col in columns] for row_dict in values_list]

                    # Construct and execute the insert query
                    insert_query = f"""
                        INSERT INTO recipes ({", ".join(columns)})
                        VALUES {values_template}
                        ON CONFLICT (recipe_id) DO UPDATE
                        SET {", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'recipe_id'])}
                    """

                    # Execute in batches
                    for row_values in values:
                        try:
                            cur.execute(insert_query, row_values)
                            inserted_count += 1
                        except Exception as e:
                            print(f"Error inserting recipe: {e}")
                            conn.rollback()

                    conn.commit()
                except Exception as e:
                    print(f"Error in batch insert: {e}")
                    conn.rollback()

                pbar.update(len(batch))

        print(f"Inserted {inserted_count} recipes")


def insert_recipe_ingredients(conn, recipes_df, ingredient_mapping):
    """Insert recipe-ingredient relationships"""
    print("Inserting recipe ingredients...")

    inserted_count = 0
    with conn.cursor() as cur:
        for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df)):
            recipe_id = row['recipe_id']
            ingredients = row.get('RecipeIngredientParts', [])
            quantities = row.get('RecipeIngredientQuantities', [])

            # Ensure quantities list is at least as long as ingredients list
            quantities.extend([None] * (len(ingredients) - len(quantities)))

            for idx, (ingredient, quantity) in enumerate(zip(ingredients, quantities)):
                if ingredient in ingredient_mapping:
                    try:
                        cur.execute(
                            """
                            INSERT INTO recipe_ingredients 
                            (recipe_id, ingredient_id, quantity, display_order)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (recipe_id, ingredient_id, display_order) DO NOTHING
                            """,
                            (recipe_id, ingredient_mapping[ingredient], quantity, idx)
                        )
                        inserted_count += 1
                    except Exception as e:
                        print(f"Error inserting ingredient {ingredient} for recipe {recipe_id}: {e}")
                        conn.rollback()

            # Commit every 100 recipes to avoid long transactions
            if (_ + 1) % 100 == 0:
                conn.commit()

        conn.commit()
    print(f"Inserted {inserted_count} recipe ingredients")


def insert_recipe_instructions(conn, recipes_df):
    """Insert recipe instructions"""
    print("Inserting recipe instructions...")

    inserted_count = 0
    with conn.cursor() as cur:
        for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df)):
            recipe_id = row['recipe_id']
            instructions = row.get('RecipeInstructions', [])

            for step_number, instruction in enumerate(instructions, 1):
                try:
                    cur.execute(
                        """
                        INSERT INTO recipe_instructions 
                        (recipe_id, step_number, instruction_text)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (recipe_id, step_number) 
                        DO UPDATE SET instruction_text = EXCLUDED.instruction_text
                        """,
                        (recipe_id, step_number, instruction)
                    )
                    inserted_count += 1
                except Exception as e:
                    print(f"Error inserting instruction {step_number} for recipe {recipe_id}: {e}")
                    conn.rollback()

            # Commit every 100 recipes to avoid long transactions
            if (_ + 1) % 100 == 0:
                conn.commit()

        conn.commit()
    print(f"Inserted {inserted_count} recipe instructions")


def insert_recipe_keywords(conn, recipes_df, keyword_mapping):
    """Insert recipe-keyword relationships"""
    print("Inserting recipe keywords...")

    inserted_count = 0
    with conn.cursor() as cur:
        for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df)):
            recipe_id = row['recipe_id']
            keywords = row.get('Keywords', [])

            for keyword in keywords:
                if keyword in keyword_mapping:
                    try:
                        cur.execute(
                            """
                            INSERT INTO recipe_keywords
                            (recipe_id, keyword_id)
                            VALUES (%s, %s)
                            ON CONFLICT (recipe_id, keyword_id) DO NOTHING
                            """,
                            (recipe_id, keyword_mapping[keyword])
                        )
                        inserted_count += 1
                    except Exception as e:
                        print(f"Error inserting keyword {keyword} for recipe {recipe_id}: {e}")
                        conn.rollback()

            # Commit every 100 recipes to avoid long transactions
            if (_ + 1) % 100 == 0:
                conn.commit()

        conn.commit()
    print(f"Inserted {inserted_count} recipe keywords")


def insert_recipe_images(conn, recipes_df):
    """Insert recipe images"""
    print("Inserting recipe images...")

    inserted_count = 0
    with conn.cursor() as cur:
        for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df)):
            recipe_id = row['recipe_id']
            images = row.get('Images', [])

            for idx, image_url in enumerate(images):
                try:
                    cur.execute(
                        """
                        INSERT INTO recipe_images
                        (recipe_id, url, is_primary)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (recipe_id, url) DO NOTHING
                        """,
                        (recipe_id, image_url, idx == 0)  # First image is primary
                    )
                    inserted_count += 1
                except Exception as e:
                    print(f"Error inserting image {image_url} for recipe {recipe_id}: {e}")
                    conn.rollback()

            # Commit every 100 recipes to avoid long transactions
            if (_ + 1) % 100 == 0:
                conn.commit()

        conn.commit()
    print(f"Inserted {inserted_count} recipe images")


def insert_reviews(conn, reviews_df):
    """Insert processed reviews into the database"""
    print("Inserting reviews into database...")

    # Create a cursor for bulk insertion
    with conn.cursor() as cur:
        # Get the list of columns in the target table
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'reviews'")
        db_columns = [row[0] for row in cur.fetchall()]

        if 'created_at' in db_columns:
            db_columns.remove('created_at')

        # Prepare review data for insertion
        inserted_count = 0

        # Use a batch size to avoid memory issues
        batch_size = 3000

        # Convert rating to numeric
        reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')

        # Option 1: Skip reviews with a rating of 0
        reviews_df = reviews_df[reviews_df['rating'] != 0]

        # Option 2: Replace a rating of 0 with None (if you prefer to treat it as missing)
        reviews_df.loc[reviews_df['rating'] == 0, 'rating'] = None

        # Convert date columns to proper datetime format
        try:
            reviews_df['date_submitted'] = pd.to_datetime(reviews_df['date_submitted'], errors='coerce')
            reviews_df['date_modified'] = pd.to_datetime(reviews_df['date_modified'], errors='coerce')
        except:
            reviews_df['date_submitted'] = None
            reviews_df['date_modified'] = None

        # Process in batches
        with tqdm(total=len(reviews_df)) as pbar:
            for i in range(0, len(reviews_df), batch_size):
                batch = reviews_df.iloc[i:i + batch_size]

                # Prepare data for this batch
                values_list = []
                for _, row in batch.iterrows():
                    # Create a dictionary of values for this row
                    row_dict = {}
                    for col in db_columns:
                        if col in row:
                            value = row[col]
                            # Handle NaN and None
                            if pd.isna(value):
                                row_dict[col] = None
                            else:
                                row_dict[col] = value
                        else:
                            row_dict[col] = None

                    values_list.append(row_dict)

                # Skip if no valid rows in this batch
                if not values_list:
                    continue

                try:
                    # Use execute_values for efficient batch insertion
                    columns = [col for col in db_columns if col in values_list[0]]
                    values_template = "(" + ", ".join(["%s"] * len(columns)) + ")"

                    # Prepare the values in the order of columns
                    values = [[row_dict.get(col) for col in columns] for row_dict in values_list]

                    # Construct and execute the insert query
                    insert_query = f"""
                        INSERT INTO reviews ({", ".join(columns)})
                        VALUES {values_template}
                        ON CONFLICT (review_id) DO UPDATE
                        SET {", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'review_id'])}
                    """

                    # Execute in batches
                    for row_values in values:
                        try:
                            cur.execute(insert_query, row_values)
                            inserted_count += 1
                        except Exception as e:
                            print(f"Error inserting review: {e}")
                            conn.rollback()

                    conn.commit()
                except Exception as e:
                    print(f"Error in batch insert: {e}")
                    conn.rollback()

                pbar.update(len(batch))

        print(f"Inserted {inserted_count} reviews")


def create_default_folders(conn, user_id):
    """Create default folders for the test user"""
    print("Creating default folders for the test user...")

    default_folders = [
        ('Favorites', 'My favorite recipes'),
        ('To Try', 'Recipes I want to try'),
        ('Quick Meals', 'Recipes that can be prepared quickly')
    ]

    with conn.cursor() as cur:
        for name, description in default_folders:
            try:
                cur.execute(
                    """
                    INSERT INTO folders (user_id, name, description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, name) DO NOTHING
                    RETURNING folder_id
                    """,
                    (user_id, name, description)
                )
                folder_id = cur.fetchone()
                if folder_id:
                    print(f"Created folder '{name}' with ID: {folder_id[0]}")
            except Exception as e:
                print(f"Error creating folder '{name}': {e}")
                conn.rollback()

    conn.commit()


def add_sample_bookmarks(conn, user_id):
    """Add some sample bookmarks for the test user"""
    print("Adding sample bookmarks for the test user...")

    # Get a few random recipe IDs
    with conn.cursor() as cur:
        cur.execute("SELECT recipe_id FROM recipes ORDER BY random() LIMIT 5")
        recipe_ids = [row[0] for row in cur.fetchall()]

        # Get the user's folder IDs
        cur.execute("SELECT folder_id, name FROM folders WHERE user_id = %s", (user_id,))
        folders = {name: folder_id for folder_id, name in cur.fetchall()}

        # Add bookmarks
        if recipe_ids and folders:
            try:
                # Add to Favorites
                for recipe_id in recipe_ids[:2]:
                    cur.execute(
                        """
                        INSERT INTO bookmarks (user_id, recipe_id, folder_id, user_rating, notes)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, recipe_id, folder_id) DO NOTHING
                        """,
                        (user_id, recipe_id, folders['Favorites'], 5, "One of my favorites!")
                    )

                # Add to To Try
                for recipe_id in recipe_ids[2:4]:
                    cur.execute(
                        """
                        INSERT INTO bookmarks (user_id, recipe_id, folder_id, notes)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, recipe_id, folder_id) DO NOTHING
                        """,
                        (user_id, recipe_id, folders['To Try'], "Looks promising")
                    )

                # Add to Quick Meals
                if len(recipe_ids) >= 5:
                    cur.execute(
                        """
                        INSERT INTO bookmarks (user_id, recipe_id, folder_id, user_rating, notes)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, recipe_id, folder_id) DO NOTHING
                        """,
                        (user_id, recipe_ids[4], folders['Quick Meals'], 4, "Great for weeknights")
                    )

                conn.commit()
                print("Added sample bookmarks successfully!")
            except Exception as e:
                print(f"Error adding sample bookmarks: {e}")
                conn.rollback()


def main():
    """Main function to load and process the data"""
    start_time = time.time()

    # Ask for CSV file paths
    recipes_path = "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/recipes.csv"
    reviews_path = "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/reviews.csv"

    # Check if files exist
    if not os.path.exists(recipes_path):
        print(f"Error: The file {recipes_path} does not exist!")
        return

    if not os.path.exists(reviews_path):
        print(f"Error: The file {reviews_path} does not exist!")
        return

    print(f"Loading data from {recipes_path} and {reviews_path}...")

    try:
        # Load datasets
        recipes_df = pd.read_csv(recipes_path, low_memory=False)
        reviews_df = pd.read_csv(reviews_path, low_memory=False)

        print(f"Loaded {len(recipes_df)} recipes and {len(reviews_df)} reviews")

        # Process data
        processed_recipes_df = process_recipes_data(recipes_df)
        processed_reviews_df = process_reviews_data(reviews_df)

        # Connect to database
        conn = connect_to_db()

        # Create a test user
        user_id = create_test_user(conn)

        # Insert nutrition information
        nutrition_mapping = insert_nutrition_info(conn, processed_recipes_df)

        # Insert authors
        insert_authors(conn, processed_recipes_df)

        # Insert keywords and get mapping
        keyword_mapping = insert_keywords(conn, processed_recipes_df)

        # Insert ingredients and get mapping
        ingredient_mapping = insert_ingredients(conn, processed_recipes_df)

        # Insert recipes FIRST - this is the key change
        insert_recipes(conn, processed_recipes_df, nutrition_mapping)

        # Check for any recipe ID discrepancies (optional)
        with conn.cursor() as cur:
            cur.execute("SELECT recipe_id FROM recipes")
            existing_recipe_ids = {row[0] for row in cur.fetchall()}

        recipe_ids_in_df = set(processed_recipes_df['recipe_id'].unique())
        missing_recipes = recipe_ids_in_df - existing_recipe_ids

        if missing_recipes:
            print(f"Warning: {len(missing_recipes)} recipes failed to insert properly")
            print(f"First few problematic IDs: {list(missing_recipes)[:5]}")

        # Insert recipe relationships AFTER recipes exist in the database
        insert_recipe_ingredients(conn, processed_recipes_df, ingredient_mapping)
        insert_recipe_instructions(conn, processed_recipes_df)
        insert_recipe_keywords(conn, processed_recipes_df, keyword_mapping)
        insert_recipe_images(conn, processed_recipes_df)

        # Insert reviews
        insert_reviews(conn, processed_reviews_df)

        # Create sample user data
        create_default_folders(conn, user_id)
        add_sample_bookmarks(conn, user_id)

        # Close connection
        conn.close()

        print(f"Data loading complete! Total time: {(time.time() - start_time):.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # This will show the full stack trace for debugging


if __name__ == "__main__":
    main()