from flask import Flask
from .routes import bp as main_bp
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB upload limit (edit as needed)
    app.register_blueprint(main_bp)
    return app
