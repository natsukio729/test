from flask import Flask
from .routes import main

def create_app():
    app = Flask(__name__)
    app.config['On202501'] = 'your_secret_key'
    app.register_blueprint(main)
    return app
