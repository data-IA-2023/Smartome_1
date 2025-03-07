# Utiliser une image Python comme base
FROM python:3.10

# Définir le répertoire de travail dans le conteneur
WORKDIR /code

# Copier les fichiers du projet dans le conteneur
COPY . /code/

RUN pip install --upgrade pip
# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application Django écoutera
EXPOSE 8000

# Commande par défaut pour démarrer l'application Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
