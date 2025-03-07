"""
URL configuration for Talocka project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Authentification import views as authviews
from Application_1 import views as app_1_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('connexion/', authviews.connexion , name='connexion'),
    path('', authviews.connexion , name='connexion'),
    path('inscription/', authviews.inscription , name='inscription'),
    path('deconnexion/', authviews.deconnexion , name='deconnexion'),
    path('accueil/', app_1_views.accueil , name='accueil'),
    path('create_projet/', app_1_views.create_projet , name='create_projet'),
    path('projets/', app_1_views.projets , name='projets'),
    path('modifier_projet/<int:projet_id>/', app_1_views.modifier_projet, name='modifier_projet'),
    path('delete_projet/', app_1_views.delete_projet , name='delete_projet'),
    path('modification/<int:projet_id>/', app_1_views.modification , name='modification'),
    path('upload_dataset/<int:projet_id>/', app_1_views.upload_dataset , name='upload_dataset'),
    path('delete_dataset/<int:dataset_id>/', app_1_views.delete_dataset, name='delete_dataset'),
    ]
