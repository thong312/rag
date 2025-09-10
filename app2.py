from flask import Flask
from routes import api_bp
from config import HOST, PORT, DEBUG

def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    return app

def start_app():
    """Start the Flask application"""
    app = create_app()
    app.run(host=HOST, port=PORT, debug=DEBUG)

if __name__ == "__main__":
    start_app()