from pymongo import MongoClient

MONGO_URI = "mongodb+srv://userdb:Hiba1234.@cluster0.k3lho.mongodb.net/userdb?retryWrites=true&w=majority"
DB_NAME = "userdb"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Collections
    users_collection = db["users"]
    farm_collection = db["farminfo"]

    # Insert sample data if collections are empty
    if not db.list_collection_names():
        users_collection.insert_one({"name": "Test User", "email": "test@example.com"})
        farm_collection.insert_one({"farm_name": "Test Farm", "location": "Test Location"})
    
    print("Databases:", client.list_database_names())
    print("Collections in 'userdb':", db.list_collection_names())
except Exception as e:
    print("Error:", e)