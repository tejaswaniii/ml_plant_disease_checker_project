from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")  # or use your cloud MongoDB URI
db = client["plant_disease_db"]
collection = db["predictions"]
