from __future__ import absolute_import, unicode_literals

# Pour s'assurer que l'application Celery est chargée dès le démarrage
from .celery import app as celery_app

__all__ = ('celery_app',)