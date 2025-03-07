from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Le nom du projet Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kaelig.settings")

# Création de l'instance Celery
app = Celery('celeri')

# Utilisation de Redis comme broker
app.config_from_object('django.conf:settings', namespace='CELERY')

# Découverte des tâches dans les apps Django
app.autodiscover_tasks()