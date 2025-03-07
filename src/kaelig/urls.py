"""
URL configuration for kaelig project.

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
from django.urls import include, path
from import_donnee import views as import_donnee_views
from auth_user import views as authviews





urlpatterns = [
    path('admin/', admin.site.urls),
    path('connexion/', authviews.connexion , name='connexion'),
    path('inscription/', authviews.inscription , name='inscription'),
    path('', authviews.connexion , name='connexion'),
    path('accueil/', import_donnee_views.accueil , name='accueil'),
    path('create_project', import_donnee_views.create_project , name='create_project'),
    path('delete_project/', import_donnee_views.delete_project, name='delete_project'),
    path('deconnexion/', authviews.deconnexion , name='deconnexion'),
    path('vosprojets/', import_donnee_views.projects , name='projects'),
    path('projet_data/', import_donnee_views.projet_data, name='projet_data'),
    path('upload_fichier/', import_donnee_views.upload_fichier, name='upload_fichier'),
    path('dataset_info/', import_donnee_views.dataset_info, name='dataset_info'),
    path('dataset_cleanning/', import_donnee_views.cleanning, name='cleanning'),
    path('imputation/', import_donnee_views.imputation, name='imputation'),
    path('visualisation/', import_donnee_views.visualisation, name='visualisation'),
    path('modelisation/', import_donnee_views.modelisation, name='modelisation'),
    
]
