from backend.mongo_db import prediction_collection

prediction_collection.insert_one({
    "test": "MongoDB Working"
})

print("Data Inserted Successfully")