from fastapi import FastAPI
from pydantic import BaseModel
from backend.orchestrator import master_agent
from backend.mongo_db import prediction_collection, chat_collection
import os

import pandas as pd
import numpy as np
import joblib

app = FastAPI(
    title="Smart Retail AI Assistant",
    description="Retail Forecasting + Analytics API",
    version="1.0"
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sales_forecast_model.pkl"
)

COLUMN_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_columns.pkl"
)

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMN_PATH)

class SalesInput(BaseModel):
    Quantity: int
    Discount: float
    Profit: float
    Order_Month: int
    Order_Year: int
    Order_Day: int
    Ship_Delay_Days: int
    Ship_Mode: str
    Segment: str
    Country: str
    City: str
    State: str
    Region: str
    Category: str
    Sub_Category: str

@app.get("/")
def home():
    return {
        "message": "Smart Retail AI Backend Running"
    }

@app.post("/predict")
def predict_sales(data: SalesInput):
    try:
        sample = {
            "Quantity": data.Quantity,
            "Discount": data.Discount,
            "Profit": data.Profit,
            "Order Month": data.Order_Month,
            "Order Year": data.Order_Year,
            "Order Day": data.Order_Day,
            "Ship Delay Days": data.Ship_Delay_Days,
            f"Ship Mode_{data.Ship_Mode}": 1,
            f"Segment_{data.Segment}": 1,
            f"Country_{data.Country}": 1,
            f"City_{data.City}": 1,
            f"State_{data.State}": 1,
            f"Region_{data.Region}": 1,
            f"Category_{data.Category}": 1,
            f"Sub-Category_{data.Sub_Category}": 1
        }

        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0

        for key, value in sample.items():
            if key in input_df.columns:
                input_df.at[0, key] = value

        prediction_log = model.predict(input_df)[0]
        prediction = np.expm1(prediction_log)
        final_prediction = float(round(prediction, 2))

        prediction_collection.insert_one({
            "quantity": data.Quantity,
            "discount": data.Discount,
            "profit": data.Profit,
            "predicted_sales": final_prediction
        })

        return {
            "Predicted Sales": final_prediction
        }

    except Exception as e:
        return {
            "error": str(e)
        }

@app.get("/analytics")
def analytics():
    try:
        DATASET_PATH = os.path.join(
            BASE_DIR,
            "dataset",
            "train.csv"
        )

        df = pd.read_csv(DATASET_PATH)
        total_sales = df["Sales"].sum()
        avg_sales = df["Sales"].mean()
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        top_region = df.groupby("Region")["Sales"].sum().idxmax()
        total_orders = len(df)

        return {
            "total_sales": float(round(total_sales, 2)),
            "average_sales": float(round(avg_sales, 2)),
            "top_category": top_category,
            "top_region": top_region,
            "total_orders": total_orders
        }

    except Exception as e:
        return {
            "error": str(e)
        }

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chatbot(request: ChatRequest):
    try:
        user_message = request.message
        response = master_agent(user_message)

        chat_collection.insert_one({
            "question": user_message,
            "response": response
        })

        return {
            "response": response
        }

    except Exception as e:
        return {
            "error": str(e)
        }

@app.get("/prediction-history")
def prediction_history():
    records = []
    for doc in prediction_collection.find():
        records.append({
            "id": str(doc["_id"]),
            "quantity": doc.get("quantity"),
            "discount": doc.get("discount"),
            "profit": doc.get("profit"),
            "predicted_sales": doc.get("predicted_sales")
        })
    return records

@app.get("/chat-history")
def chat_history():
    chats = []
    for doc in chat_collection.find():
        chats.append({
            "id": str(doc["_id"]),
            "question": doc.get("question"),
            "response": doc.get("response")
        })
    return chats