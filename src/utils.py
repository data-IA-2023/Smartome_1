from pymongo import MongoClient
def get_db_mongo(db_name="Auto_ML", host="localhost", port=27017, username=None, password=None):
    client = MongoClient(host=host,
                        port=int(port),
                        username=username,
                        password=password
                        )
    db_mongo = client[db_name]
    col = db_mongo["User"]
    return col, client