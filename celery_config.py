import os
from celery import Celery

# Get the broker URL from the environment
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=CELERY_BROKER_URL,  # You can also use a different backend if needed
        broker=CELERY_BROKER_URL
    )
    celery.conf.update(app.config)
    return celery
