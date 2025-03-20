import concurrent.futures
import csv
from io import StringIO

import pandas as pd
import re
import os

import psutil
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

PARALLEL_WORKERS = mp.cpu_count()


def parse_r_list(text):
    """Optimized parser for R-style lists and various string formats"""
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()

    if text.startswith('c(') and text.endswith(')'):
        content = text[2:-1].strip()
        # Use regex to split on commas outside quotes
        pattern = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
        items = []
        current = ''
        in_quotes = False
        for char in content:
            if char == '"' and not in_quotes:
                in_quotes = True
            elif char == '"' and in_quotes:
                in_quotes = False
            elif char == ',' and not in_quotes:
                items.append(current.strip().strip('"'))
                current = ''
                continue
            current += char
        if current:
            items.append(current.strip().strip('"'))
        return [item for item in items if item]

    # Fallback for other formats
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                return parsed_json
        except:
            pass

    if '\n' in text:
        return [line.strip().strip('"\'') for line in text.split('\n') if line.strip()]

    if ',' in text and 'http' not in text.lower():
        return [item.strip().strip('"\'') for item in text.split(',') if item.strip()]

    if 'http' in text.lower():
        url_pattern = re.compile(
            r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        urls = [url.rstrip('",') for url in url_pattern.findall(text)]
        if urls:
            return urls

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return [text[1:-1]]

    return [text]



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


def process_dataframe_parallel(df, columns_to_process, num_workers=None):
    """Process dataframe using thread pool for better performance with I/O operations"""
    if num_workers is None:
        num_workers = min(PARALLEL_WORKERS, len(df))

    process_func = partial(process_row, columns_to_process=columns_to_process)

    chunks = np.array_split(df, num_workers)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(lambda chunk: chunk.apply(process_func, axis=1), chunk) for chunk in chunks]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return pd.concat(results)


def connect_to_db():
    """Connect to PostgreSQL database with enhanced connection parameters"""
    try:
        conn = psycopg2.connect(
            **DB_CONFIG,
            # Enhanced connection pooling parameters
            application_name='data_loader',
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
            # Set client_encoding for better text handling
            client_encoding='UTF8'
        )
        conn.set_session(autocommit=False)

        # Increase work_mem for better sorting/join performance
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '64MB'")
            cur.execute("SET maintenance_work_mem = '256MB'")

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
                conn.commit()
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
        'RecipeCategory': 'category',
        'Calories': 'calories',
        'FatContent': 'fat_content',
        'SaturatedFatContent': 'saturated_fat_content',
        'CholesterolContent': 'cholesterol_content',
        'SodiumContent': 'sodium_content',
        'CarbohydrateContent': 'carbohydrate_content',
        'FiberContent': 'fiber_content',
        'SugarContent': 'sugar_content',
        'ProteinContent': 'protein_content',
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

def insert_categories(conn, recipes_df):
    """Extract unique categories and insert them using optimized batch operations"""
    print("Inserting recipe categories...")

    # Extract all unique categories from recipes (may include multiple per recipe)
    all_categories = set()
    for _, row in recipes_df.iterrows():
        category = row.get('category')
        if isinstance(category, str) and category.strip():
            # Split by commas if multiple categories in one field
            categories = [cat.strip() for cat in category.split(',')]
            all_categories.update(categories)

    if not all_categories:
        return {}

    # Prepare data for copy operation
    category_data = StringIO()
    for category in all_categories:
        safe_category = category.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        category_data.write(f"{safe_category}\n")
    category_data.seek(0)

    with conn.cursor() as cur:
        try:
            # Create categories table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    category_id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL
                )
            """)

            # Use COPY for maximum performance
            cur.execute("CREATE TEMP TABLE tmp_categories (name VARCHAR(255) NOT NULL) ON COMMIT DROP")
            cur.copy_from(category_data, 'tmp_categories', columns=('name',))

            # Insert from temp table to actual table (handling duplicates)
            cur.execute("""
                INSERT INTO categories (name)
                SELECT DISTINCT name FROM tmp_categories
                ON CONFLICT (name) DO NOTHING
            """)

            # Get the mapping of category names to IDs
            cur.execute("SELECT name, category_id FROM categories WHERE name IN %s", (tuple(all_categories),))
            category_mapping = {name: cid for name, cid in cur.fetchall()}

            conn.commit()
            print(f"Inserted {len(category_mapping)} unique categories")
            return category_mapping
        except Exception as e:
            print(f"Error in bulk category insertion: {e}")
            conn.rollback()
            return {}


def insert_recipe_categories(conn, recipes_df, category_mapping):
    """Insert recipe-category relationships using optimized batching"""
    print("Inserting recipe categories relationships...")

    # Create the junction table if it doesn't exist
    with conn.cursor() as cur:
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recipe_categories (
                    recipe_id BIGINT REFERENCES recipes(recipe_id) ON DELETE CASCADE,
                    category_id INTEGER REFERENCES categories(category_id) ON DELETE CASCADE,
                    PRIMARY KEY (recipe_id, category_id)
                )
            """)
            conn.commit()
        except Exception as e:
            print(f"Error creating recipe_categories table: {e}")
            conn.rollback()
            return

    # Prepare all data first
    data = []
    for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Preparing category data"):
        recipe_id = row['recipe_id']
        category_str = row.get('category')

        if isinstance(category_str, str) and category_str.strip():
            categories = [cat.strip() for cat in category_str.split(',')]

            for category in categories:
                if category in category_mapping:
                    data.append((
                        recipe_id,
                        category_mapping[category]
                    ))

    # Insert in batches
    with conn.cursor() as cur:
        try:
            insert_query = """
                INSERT INTO recipe_categories
                (recipe_id, category_id)
                VALUES (%s, %s)
                ON CONFLICT (recipe_id, category_id) DO NOTHING
            """

            # Use execute_batch for better performance
            with tqdm(total=len(data), desc="Inserting category relationships") as pbar:
                for i in range(0, len(data), BATCH_SIZE):
                    batch = data[i:i + BATCH_SIZE]
                    psycopg2.extras.execute_batch(cur, insert_query, batch, page_size=BATCH_SIZE)
                    pbar.update(len(batch))

            conn.commit()
            print(f"Inserted {len(data)} recipe-category relationships")
        except Exception as e:
            print(f"Error in batch category relationship insertion: {e}")
            conn.rollback()


def insert_nutrition_info(conn, recipes_df):
    """Insert nutrition information and return a mapping of recipe_id to nutrition_id"""
    print("Inserting nutrition information...")
    nutrition_mapping = {}

    # List the nutrition columns (make sure these match your CSV)
    nutrition_columns = [
        'calories', 'fat_content', 'saturated_fat_content',
        'cholesterol_content', 'sodium_content', 'carbohydrate_content',
        'fiber_content', 'sugar_content', 'protein_content'
    ]

    # Prepare nutrition data
    nutrition_data = []
    recipe_ids = []

    # Use pd.notna to check for valid (non-NaN) nutrition values
    for idx, row in recipes_df.iterrows():
        nutrition_values = [row.get(col) for col in nutrition_columns]
        if any(pd.notna(v) for v in nutrition_values):
            nutrition_data.append(nutrition_values)
            recipe_ids.append(row['recipe_id'])

    if not nutrition_data:
        print("No nutrition data found.")
        return nutrition_mapping

    with conn.cursor() as cur:
        try:
            # Build the INSERT query for nutrition_info
            query = f"""
                INSERT INTO nutrition_info (
                    calories, fat_content, saturated_fat_content, 
                    cholesterol_content, sodium_content, carbohydrate_content,
                    fiber_content, sugar_content, protein_content
                )
                VALUES ({', '.join(['%s'] * len(nutrition_columns))})
                RETURNING nutrition_id
            """
            # Loop over each row and insert individually
            for i, values in enumerate(nutrition_data):
                cur.execute(query, values)
                nutrition_id = cur.fetchone()[0]
                # Map the recipe_id to its corresponding nutrition_id
                nutrition_mapping[recipe_ids[i]] = nutrition_id

            conn.commit()
            print(f"Inserted {len(nutrition_mapping)} nutrition records")
            return nutrition_mapping

        except Exception as e:
            print(f"Error inserting nutrition info: {e}")
            conn.rollback()
            return {}


def insert_authors(conn, recipes_df):
    """Insert authors using optimized batch operations"""
    print("Inserting authors...")

    # Extract unique authors
    unique_authors = recipes_df[['author_id', 'author_name']].drop_duplicates()
    unique_authors = unique_authors[~unique_authors['author_id'].isna()]

    if unique_authors.empty:
        return {}

    with conn.cursor() as cur:
        try:
            # Prepare data for bulk insertion
            data = [(row['author_id'], row['author_name']) for _, row in unique_authors.iterrows()]

            # Use execute_batch for better performance
            insert_query = """
                INSERT INTO authors (author_id, name)
                VALUES (%s, %s)
                ON CONFLICT (author_id) DO UPDATE
                SET name = EXCLUDED.name
            """
            psycopg2.extras.execute_batch(cur, insert_query, data, page_size=BATCH_SIZE)
            conn.commit()
            print(f"Inserted/updated {len(data)} authors")
            return True
        except Exception as e:
            print(f"Error batch inserting authors: {e}")
            conn.rollback()
            return {}


def insert_keywords(conn, recipes_df):
    """Insert keywords using optimized bulk operations"""
    print("Inserting keywords...")

    # Extract all unique keywords from recipes
    all_keywords = set()
    for _, row in recipes_df.iterrows():
        if 'Keywords' in row and isinstance(row['Keywords'], list):
            all_keywords.update(row['Keywords'])

    if not all_keywords:
        return {}

    # Prepare data for copy operation
    keyword_data = StringIO()
    for keyword in all_keywords:
        keyword_data.write(f"{keyword}\n")
    keyword_data.seek(0)

    with conn.cursor() as cur:
        try:
            # Use COPY for maximum performance
            cur.execute("CREATE TEMP TABLE tmp_keywords (name VARCHAR(255) NOT NULL) ON COMMIT DROP")
            cur.copy_from(keyword_data, 'tmp_keywords', columns=('name',))

            # Insert from temp table to actual table (handling duplicates)
            cur.execute("""
                INSERT INTO keywords (name)
                SELECT DISTINCT name FROM tmp_keywords
                ON CONFLICT (name) DO NOTHING
            """)

            # Get the mapping
            cur.execute("SELECT name, keyword_id FROM keywords WHERE name IN %s", (tuple(all_keywords),))
            keyword_mapping = {name: kid for name, kid in cur.fetchall()}

            conn.commit()
            print(f"Inserted {len(keyword_mapping)} unique keywords")
            return keyword_mapping
        except Exception as e:
            print(f"Error in bulk keyword insertion: {e}")
            conn.rollback()
            return {}


def insert_ingredients(conn, recipes_df):
    """Insert ingredients using optimized bulk operations"""
    print("Inserting ingredients...")

    # Extract all unique ingredients from recipes
    all_ingredients = set()
    for _, row in recipes_df.iterrows():
        if 'RecipeIngredientParts' in row and isinstance(row['RecipeIngredientParts'], list):
            all_ingredients.update(row['RecipeIngredientParts'])

    if not all_ingredients:
        return {}

    # Prepare data for copy operation
    ingredient_data = StringIO()
    for ingredient in all_ingredients:
        # Escape special characters like tabs, newlines, etc.
        safe_ingredient = ingredient.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        ingredient_data.write(f"{safe_ingredient}\n")
    ingredient_data.seek(0)

    with conn.cursor() as cur:
        try:
            # Use COPY for maximum performance
            cur.execute("CREATE TEMP TABLE tmp_ingredients (name VARCHAR(255) NOT NULL) ON COMMIT DROP")
            cur.copy_from(ingredient_data, 'tmp_ingredients', columns=('name',))

            # Insert from temp table to actual table (handling duplicates)
            cur.execute("""
                INSERT INTO ingredients (name)
                SELECT DISTINCT name FROM tmp_ingredients
                ON CONFLICT (name) DO NOTHING
            """)

            # Get the mapping
            cur.execute("SELECT name, ingredient_id FROM ingredients WHERE name IN %s", (tuple(all_ingredients),))
            ingredient_mapping = {name: iid for name, iid in cur.fetchall()}

            conn.commit()
            print(f"Inserted {len(ingredient_mapping)} unique ingredients")
            return ingredient_mapping
        except Exception as e:
            print(f"Error in bulk ingredient insertion: {e}")
            conn.rollback()


def insert_recipes(conn, recipes_df, nutrition_mapping):
    """Insert processed recipes in optimized batches"""
    print("Inserting recipes into database...")
    successful_recipe_ids = set()

    # Create a cursor for bulk insertion
    with conn.cursor() as cur:
        # Get the list of columns in the target table
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'recipes'")
        db_columns = [row[0] for row in cur.fetchall()]

        # Remove auto-generated columns
        excludes = ['created_at', 'search_vector']
        db_columns = [col for col in db_columns if col not in excludes]

        # Convert DataFrame column types to match PostgreSQL expectations
        recipes_df['aggregated_rating'] = pd.to_numeric(recipes_df['aggregated_rating'], errors='coerce')
        recipes_df['review_count'] = pd.to_numeric(recipes_df['review_count'], errors='coerce')
        recipes_df['recipe_servings'] = pd.to_numeric(recipes_df['recipe_servings'], errors='coerce')

        # Convert date columns to proper datetime format
        recipes_df['date_published'] = pd.to_datetime(recipes_df['date_published'], errors='coerce')

        # Add nutrition_id from mapping
        recipes_df['nutrition_id'] = recipes_df['recipe_id'].map(nutrition_mapping)

        problematic_ids = []

        # Process in batches
        with tqdm(total=len(recipes_df)) as pbar:
            for i in range(0, len(recipes_df), BATCH_SIZE):
                batch = recipes_df.iloc[i:i + BATCH_SIZE]

                # Prepare data for this batch
                batch_data = []
                for _, row in batch.iterrows():
                    # Create a list of values for this row
                    values = []
                    for col in db_columns:
                        if col in row:
                            value = row[col]
                            # Handle NaN, None, and special data types
                            if pd.isna(value):
                                values.append(None)
                            else:
                                values.append(value)
                        else:
                            values.append(None)
                    batch_data.append(values)

                # Skip if no valid rows in this batch
                if not batch_data:
                    continue

                try:
                    # Construct the insert query
                    columns_str = ", ".join(db_columns)
                    placeholders = ", ".join(["%s"] * len(db_columns))
                    update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in db_columns if col != 'recipe_id'])

                    insert_query = f"""
                        INSERT INTO recipes ({columns_str})
                        VALUES ({placeholders})
                        ON CONFLICT (recipe_id) DO UPDATE
                        SET {update_str}
                    """

                    # Use execute_batch for better performance
                    psycopg2.extras.execute_batch(cur, insert_query, batch_data, page_size=BATCH_SIZE)
                    conn.commit()

                    for _, row in batch.iterrows():
                        successful_recipe_ids.add(row['recipe_id'])

                except Exception as e:
                    print(f"Error in batch insert: {e}")
                    # Log the recipe IDs in this batch
                    for _, row in batch.iterrows():
                        problematic_ids.append(row['recipe_id'])
                    conn.rollback()

                pbar.update(len(batch))
        if problematic_ids:
            print(f"Problematic recipe IDs: {problematic_ids[:10]}...")

        print(f"Inserted/updated recipes from {len(recipes_df)} records")
        return successful_recipe_ids


def insert_recipe_ingredients(conn, recipes_df, ingredient_mapping):
    """Insert recipe-ingredient relationships using optimized batching"""
    print("Inserting recipe ingredients...")

    # Prepare all data first
    data = []
    for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Preparing ingredient data"):
        recipe_id = row['recipe_id']
        ingredients = row.get('RecipeIngredientParts', [])
        quantities = row.get('RecipeIngredientQuantities', [])

        # Ensure quantities list is at least as long as ingredients list
        quantities.extend([None] * (len(ingredients) - len(quantities)))

        for idx, (ingredient, quantity) in enumerate(zip(ingredients, quantities)):
            if ingredient in ingredient_mapping:
                data.append((
                    recipe_id,
                    ingredient_mapping[ingredient],
                    quantity,
                    idx
                ))

    # Insert in batches
    with conn.cursor() as cur:
        try:
            insert_query = """
                INSERT INTO recipe_ingredients 
                (recipe_id, ingredient_id, quantity, display_order)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (recipe_id, ingredient_id, display_order) DO NOTHING
            """

            # Use execute_batch for better performance
            with tqdm(total=len(data), desc="Inserting ingredients") as pbar:
                for i in range(0, len(data), BATCH_SIZE):
                    batch = data[i:i + BATCH_SIZE]
                    psycopg2.extras.execute_batch(cur, insert_query, batch, page_size=BATCH_SIZE)
                    pbar.update(len(batch))

            conn.commit()
            print(f"Inserted {len(data)} recipe ingredients")
        except Exception as e:
            print(f"Error in batch ingredient insertion: {e}")
            conn.rollback()


def insert_recipe_instructions(conn, recipes_df):
    """Insert recipe instructions using optimized batching"""
    print("Inserting recipe instructions...")

    # Prepare all data first
    data = []
    for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Preparing instruction data"):
        recipe_id = row['recipe_id']
        instructions = row.get('RecipeInstructions', [])

        for step_number, instruction in enumerate(instructions, 1):
            data.append((
                recipe_id,
                step_number,
                instruction
            ))

    # Insert in batches
    with conn.cursor() as cur:
        try:
            insert_query = """
                INSERT INTO recipe_instructions 
                (recipe_id, step_number, instruction_text)
                VALUES (%s, %s, %s)
                ON CONFLICT (recipe_id, step_number) 
                DO UPDATE SET instruction_text = EXCLUDED.instruction_text
            """

            # Use execute_batch for better performance
            with tqdm(total=len(data), desc="Inserting instructions") as pbar:
                for i in range(0, len(data), BATCH_SIZE):
                    batch = data[i:i + BATCH_SIZE]
                    psycopg2.extras.execute_batch(cur, insert_query, batch, page_size=BATCH_SIZE)
                    pbar.update(len(batch))

            conn.commit()
            print(f"Inserted {len(data)} recipe instructions")
        except Exception as e:
            print(f"Error in batch instruction insertion: {e}")
            conn.rollback()


def insert_recipe_keywords(conn, recipes_df, keyword_mapping):
    """Insert recipe-keyword relationships using optimized batching"""
    print("Inserting recipe keywords...")

    # Prepare all data first
    data = []
    for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Preparing keyword data"):
        recipe_id = row['recipe_id']
        keywords = row.get('Keywords', [])

        for keyword in keywords:
            if keyword in keyword_mapping:
                data.append((
                    recipe_id,
                    keyword_mapping[keyword]
                ))

    # Insert in batches
    with conn.cursor() as cur:
        try:
            insert_query = """
                INSERT INTO recipe_keywords
                (recipe_id, keyword_id)
                VALUES (%s, %s)
                ON CONFLICT (recipe_id, keyword_id) DO NOTHING
            """

            # Use execute_batch for better performance
            with tqdm(total=len(data), desc="Inserting keywords") as pbar:
                for i in range(0, len(data), BATCH_SIZE):
                    batch = data[i:i + BATCH_SIZE]
                    psycopg2.extras.execute_batch(cur, insert_query, batch, page_size=BATCH_SIZE)
                    pbar.update(len(batch))

            conn.commit()
            print(f"Inserted {len(data)} recipe keywords")
        except Exception as e:
            print(f"Error in batch keyword insertion: {e}")
            conn.rollback()


def insert_recipe_images(conn, recipes_df):
    """Insert recipe images using optimized batching"""
    print("Inserting recipe images...")

    # Prepare all data first
    data = []
    for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Preparing image data"):
        recipe_id = row['recipe_id']
        images = row.get('Images', [])

        for idx, image_url in enumerate(images):
            if image_url.startswith("http"):
                data.append((
                    recipe_id,
                    image_url,
                    idx == 0  # First image is primary
                ))

    # Insert in batches
    with conn.cursor() as cur:
        try:
            insert_query = """
                INSERT INTO recipe_images
                (recipe_id, url, is_primary)
                VALUES (%s, %s, %s)
                ON CONFLICT (recipe_id, url) DO NOTHING
            """

            # Use execute_batch for better performance
            with tqdm(total=len(data), desc="Inserting images") as pbar:
                for i in range(0, len(data), BATCH_SIZE):
                    batch = data[i:i + BATCH_SIZE]
                    psycopg2.extras.execute_batch(cur, insert_query, batch, page_size=BATCH_SIZE)
                    pbar.update(len(batch))

            conn.commit()
            print(f"Inserted {len(data)} recipe images")
        except Exception as e:
            print(f"Error in batch image insertion: {e}")
            conn.rollback()


def insert_reviews(conn, reviews_df):
    """Insert processed reviews into the database using optimized batch operations"""
    print("Inserting reviews into database...")

    # Create a cursor for batch insertion
    with conn.cursor() as cur:
        # Get the list of columns in the target table
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'reviews'")
        db_columns = [row[0] for row in cur.fetchall()]

        # Remove auto-generated columns
        if 'created_at' in db_columns:
            db_columns.remove('created_at')

        # Convert rating to numeric and handle special cases
        reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')

        # Allow rating = 0
        reviews_df = reviews_df[pd.notna(reviews_df['rating'])]

        # Convert date columns to proper datetime format
        try:
            reviews_df['date_submitted'] = pd.to_datetime(reviews_df['date_submitted'], errors='coerce')
            reviews_df['date_modified'] = pd.to_datetime(reviews_df['date_modified'], errors='coerce')
        except:
            print("Error converting date columns, setting to None")
            reviews_df['date_submitted'] = None
            reviews_df['date_modified'] = None

        # Process in batches with better tracking
        total_inserted = 0
        with tqdm(total=len(reviews_df), desc="Inserting reviews") as pbar:
            for i in range(0, len(reviews_df), BATCH_SIZE):
                batch = reviews_df.iloc[i:i + BATCH_SIZE]

                # Prepare data for this batch
                batch_data = []
                for _, row in batch.iterrows():
                    # Create a list of values for this row in the correct column order
                    values = []
                    for col in db_columns:
                        if col in row:
                            value = row[col]
                            # Handle NaN and None
                            if pd.isna(value):
                                values.append(None)
                            else:
                                values.append(value)
                        else:
                            values.append(None)
                    batch_data.append(values)

                # Skip if no valid rows in this batch
                if not batch_data:
                    continue

                try:
                    # Construct the insert query
                    columns_str = ", ".join(db_columns)
                    placeholders = ", ".join(["%s"] * len(db_columns))
                    update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in db_columns if col != 'review_id'])

                    insert_query = f"""
                        INSERT INTO reviews ({columns_str})
                        VALUES ({placeholders})
                        ON CONFLICT (review_id) DO UPDATE
                        SET {update_str}
                    """

                    # Use execute_batch for better performance
                    psycopg2.extras.execute_batch(cur, insert_query, batch_data, page_size=BATCH_SIZE)
                    conn.commit()
                    total_inserted += len(batch)
                except Exception as e:
                    print(f"Error in batch insert: {e}")
                    conn.rollback()

                pbar.update(len(batch))

        print(f"Successfully inserted/updated {total_inserted} reviews")


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
    start_time = time.time()

    # Configure batch size based on system memory
    global BATCH_SIZE
    # Use smaller batch size for lower memory systems
    system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # in GB
    if system_memory < 8:
        BATCH_SIZE = 1000
    elif system_memory < 16:
        BATCH_SIZE = 5000
    else:
        BATCH_SIZE = 10000

    # File paths
    recipes_path = "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/recipes.csv"
    reviews_path = "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/reviews.csv"

    if not os.path.exists(recipes_path):
        print(f"Error: The file {recipes_path} does not exist!")
        return

    if not os.path.exists(reviews_path):
        print(f"Error: The file {reviews_path} does not exist!")
        return

    print(f"Loading data from {recipes_path} and {reviews_path}...")

    try:
        # Read CSV with optimized settings
        recipes_df = pd.read_csv(recipes_path, low_memory=False, dtype={'RecipeId': 'int64'})
        reviews_df = pd.read_csv(reviews_path, low_memory=False, dtype={'ReviewId': 'int64', 'RecipeId': 'int64'})

        print(f"Loaded {len(recipes_df)} recipes and {len(reviews_df)} reviews")

        processed_recipes_df = process_recipes_data(recipes_df)
        processed_reviews_df = process_reviews_data(reviews_df)

        # Ensure recipe_id is integer
        processed_recipes_df['recipe_id'] = processed_recipes_df['recipe_id'].astype(int)

        conn = connect_to_db()
        user_id = create_test_user(conn)

        # Insert data in logical order
        nutrition_mapping = insert_nutrition_info(conn, processed_recipes_df)

        # Insert authors from both recipes and reviews to avoid missing foreign key issues
        insert_authors(conn, processed_recipes_df)
        insert_authors(conn, processed_reviews_df)

        # Process categories (new functionality)
        category_mapping = insert_categories(conn, processed_recipes_df)

        # Process keywords and ingredients
        keyword_mapping = insert_keywords(conn, processed_recipes_df)
        ingredient_mapping = insert_ingredients(conn, processed_recipes_df)

        # Map nutrition IDs to recipes
        processed_recipes_df['nutrition_id'] = processed_recipes_df['recipe_id'].map(nutrition_mapping)

        # Insert recipes
        successful_recipe_ids = insert_recipes(conn, processed_recipes_df, nutrition_mapping)

        # Filter to only include successful recipes
        successful_recipes_df = processed_recipes_df[processed_recipes_df['recipe_id'].isin(successful_recipe_ids)]

        # Insert recipe relationships
        insert_recipe_ingredients(conn, successful_recipes_df, ingredient_mapping)
        insert_recipe_instructions(conn, successful_recipes_df)
        insert_recipe_keywords(conn, successful_recipes_df, keyword_mapping)
        insert_recipe_images(conn, successful_recipes_df)

        # Insert category relationships (new functionality)
        insert_recipe_categories(conn, successful_recipes_df, category_mapping)

        # Filter reviews to only those with existing recipes
        existing_recipe_ids = set(successful_recipes_df['recipe_id'].unique())
        filtered_reviews_df = processed_reviews_df[processed_reviews_df['recipe_id'].isin(existing_recipe_ids)]
        print(f"Filtered reviews count: {len(filtered_reviews_df)}")

        # Insert reviews and user data
        insert_reviews(conn, filtered_reviews_df)
        create_default_folders(conn, user_id)
        add_sample_bookmarks(conn, user_id)

        # Update search vectors for full text search
        with conn.cursor() as cur:
            print("Updating search vectors...")
            cur.execute(
                "UPDATE recipes SET search_vector = to_tsvector('english', coalesce(name,'') || ' ' || coalesce(description,''))")
            conn.commit()

        conn.close()
        print(f"Data loading complete! Total time: {(time.time() - start_time):.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()