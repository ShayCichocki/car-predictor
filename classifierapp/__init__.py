import os
import logging

from flask import Flask, render_template, abort

def create_app(db_uri='any'):
    app = Flask(__name__)

    # Register Blueprints
    #app.register_blueprint(main_blueprint)


    @app.teardown_request
    def shutdown_session(exception=None):
        pass

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html')

    @app.errorhandler(500)
    def internal_error(exception):
        app.logger.exception(exception)
        return "Some Internal error has taken place."

    @app.route('/')
    def hello_world():
        return 'Hello World!'


    return app