import datetime

from flask import Flask
from .views.views import page
from .views.views_img import img_page
from .views.views_user import user_center
from .views.views_admin import admin_page
from .exts import init_exts


def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint=page)
    app.register_blueprint(blueprint=img_page)
    app.register_blueprint(blueprint=user_center)
    app.register_blueprint(blueprint=admin_page)
    app.config['SECRET_KEY'] = 'abc123'
    app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)
    db_uri = 'sqlite:///sqlite3.db'
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    init_exts(app=app)
    # print(app.config)
    return app

