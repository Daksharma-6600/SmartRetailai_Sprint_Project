from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URI")
)

db = client["smart_retail_ai"]

prediction_collection = db["predictions"]

chat_collection = db["chats"]

print("MongoDB Connected Successfully")