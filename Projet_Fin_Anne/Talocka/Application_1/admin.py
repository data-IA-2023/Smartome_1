from django.contrib import admin

# Register your models here.
from .models import Projet_User

# Enregistrer le modèle dans l'interface d'administration
admin.site.register(Projet_User)