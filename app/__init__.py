from flask import Flask
from flask_cors import CORS
from app.extensions import db, jwt, migrate, cache
import os
from dotenv import load_dotenv

from .api import auth_bp
from .api import search_bp

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY'),
        JWT_TOKEN_LOCATION=["headers"],  # Ensure JWT is read from Authorization header
        JWT_HEADER_NAME="Authorization",  # Default is "Authorization"
        JWT_HEADER_TYPE="Bearer",  # Default is "Bearer"
    )

    CORS(app, supports_credentials=True)
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    cache.init_app(app)

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(search_bp, url_prefix='/api/search')


    return app