import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
import pickle
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from app.extensions import db
from app.models import Review, Recipe

# Constants
MODEL_PATH = 'ltr_model_categories_review_count.pkl'
N_TRIALS = 200
N_JOBS = 2
NUM_THREADS = os.cpu_count()
DATABASE_URI = os.getenv('DATABASE_URL')

# Initialize database
def init_db():
    if not DATABASE_URI:
        raise ValueError("DATABASE_URL not set in environment variables")
    engine = create_engine(DATABASE_URI)
    db_session = SQLAlchemy().sessionmaker(bind=engine)()
    return db_session

# Load and preprocess data
def load_and_preprocess_data(db_session):
    reviews = db_session.query(
        Review.author_id,
        Review.recipe_id,
        Review.rating
    ).all()
    df_reviews = pd.DataFrame(reviews, columns=["author_id", "recipe_id", "rating"])

    recipes = db_session.query(
        Recipe.recipe_id,
        Recipe.aggregated_rating,
        Recipe.category,
        Recipe.review_count
    ).all()
    df_recipes = pd.DataFrame(recipes, columns=["recipe_id", "aggregated_rating", "category", "review_count"])

    df = df_reviews.merge(df_recipes, on="recipe_id", how="inner")
    df_categories = pd.get_dummies(df['category'], prefix="cat")
    df = pd.concat([df, df_categories], axis=1)

    df['review_count'] = df['review_count'].fillna(0)  # Handle nulls
    feature_cols = [col for col in df.columns if col.startswith("cat_")] + ["aggregated_rating", "review_count"]
    return df, feature_cols

# Objective function (unchanged)
def objective(trial, X_train, X_test, y_train, y_test, train_groups, test_groups):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'num_threads': NUM_THREADS
    }

    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    test_data = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    return model.best_score['valid_0']['ndcg@5']

# Split data (unchanged)
def split_data(df, feature_cols):
    X = df[feature_cols]
    y = df['rating']
    groups = df['author_id']

    unique_users = df['author_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    train_df = df[df['author_id'].isin(train_users)]
    test_df = df[df['author_id'].isin(test_users)]

    X_train = train_df[feature_cols]
    y_train = train_df['rating']
    train_groups = train_df.groupby('author_id').size().tolist()

    X_test = test_df[feature_cols]
    y_test = test_df['rating']
    test_groups = test_df.groupby('author_id').size().tolist()

    return X_train, X_test, y_train, y_test, train_groups, test_groups

# Main training function
def train_and_save_best_model():
    db_session = init_db()
    df, feature_cols = load_and_preprocess_data(db_session)
    X_train, X_test, y_train, y_test, train_groups, test_groups = split_data(df, feature_cols)
    db_session.close()

    study = optuna.create_study(direction='maximize')
    print(f"Starting hyperparameter optimization with {N_TRIALS} trials using {NUM_THREADS} cores...")

    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test, train_groups, test_groups),
                   n_trials=N_TRIALS, n_jobs=N_JOBS)

    best_params = study.best_params
    print(f"Best hyperparameters found: {best_params}")

    final_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves'],
        'min_child_samples': best_params['min_child_samples'],
        'max_depth': best_params['max_depth'],
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'num_threads': NUM_THREADS
    }

    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    test_data = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=train_data)

    final_model = lgb.train(
        final_params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    print(f"Train NDCG@5: {final_model.best_score['train']['ndcg@5']:.4f}")
    print(f"Valid NDCG@5: {final_model.best_score['valid']['ndcg@5']:.4f}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_best_model()