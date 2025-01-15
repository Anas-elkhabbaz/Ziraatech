from pymongo import MongoClient

# Replace with your MongoDB connection details
MONGO_URI = "mongodb+srv://userdb:Hiba1234.@cluster0.k3lho.mongodb.net/userdb?retryWrites=true&w=majority"
DB_NAME = "userdb"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
farm_collection = db["farminfo"]

def add_farm_data(farm_name, location, username):
    """
    Adds a farm to the database for a specific user.
    """
    farm_collection.insert_one({"farm_name": farm_name, "location": location, "username": username})

def get_all_farms_by_user(username):
    """
    Retrieves all farms belonging to a specific user.
    """
    return list(farm_collection.find({"username": username}, {"_id": 0}))