from backend.db import users_collection

def authenticate_user(username, password):
    """Check if user credentials are valid."""
    user = users_collection.find_one({"name": username, "password": password})
    return user is not None

def register_user(username, password, email):
    """Register a new user."""
    if users_collection.find_one({"name": username}):
        return False  # User already exists
    users_collection.insert_one({"name": username, "password": password, "email": email})
    return True