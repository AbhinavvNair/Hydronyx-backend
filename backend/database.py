import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hydroai_db")

class Database:
    client: Optional[MongoClient] = None
    db = None

    @classmethod
    def connect_db(cls):
        """Connect to MongoDB"""
        try:
            cls.client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
            # Test connection
            cls.client.admin.command('ping')
            cls.db = cls.client[DATABASE_NAME]
            print("[OK] Connected to MongoDB successfully")
            return cls.db
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"[ERROR] Failed to connect to MongoDB: {e}")
            raise

    @classmethod
    def close_db(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            print("[OK] Disconnected from MongoDB")

    @classmethod
    def get_db(cls):
        """Get database instance"""
        if cls.db is None:
            cls.connect_db()
        return cls.db


def get_users_collection():
    """Get users collection"""
    db = Database.get_db()
    return db['users']


def get_predictions_collection():
    """Get predictions collection"""
    db = Database.get_db()
    return db['predictions']


def get_forecast_collection():
    """Get forecast_history collection"""
    db = Database.get_db()
    return db['forecast_history']


def create_indexes():
    """Create database indexes"""
    users = get_users_collection()
    users.create_index("email", unique=True)

    policy_sims = get_policy_simulations_collection()
    policy_sims.create_index([("user_id", 1), ("created_at", -1)])

    print("[OK] Database indexes created")


def get_validation_runs_collection():
    db = Database.get_db()
    return db['validation_runs']


def get_policy_simulations_collection():
    """Collection for stored policy intervention summaries."""
    db = Database.get_db()
    return db['policy_simulations']
