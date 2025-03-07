import pymongo
from gridfs import GridFS
from django.conf import settings

def get_mongo_gridfs():
    """
    Retourne une instance de GridFS pour interagir avec MongoDB.
    """
    client = pymongo.MongoClient(settings.MONGO_URI)  # Utilise l'URI de connexion défini dans settings
    db = client[settings.MONGO_DB_NAME]  # La base de données MongoDB
    grid_fs = GridFS(db)
    return db,grid_fs