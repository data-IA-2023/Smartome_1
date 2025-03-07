from mongoengine import connect

MONGO_DB_NAME = "my_mongodb"
MONGO_USER = "mongo"
MONGO_PASSWORD = "mongo"
MONGO_HOST = "mongodb" 
MONGO_PORT = 27017

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}?authSource=admin"

connect(db=MONGO_DB_NAME, host=MONGO_URI)