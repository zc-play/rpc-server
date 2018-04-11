# coding: utf-8
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.from_object('web.settings')
db = SQLAlchemy()
db.init_app(app)


app.app_context().push()

