from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from backend.main import app

client = TestClient(app)

# =========================
# HOME ENDPOINT
# =========================

def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {

        "message":
        "Smart Retail AI Backend Running"
    }


# =========================
# PREDICT ENDPOINT
# =========================

@patch("backend.main.prediction_collection")
@patch("backend.main.model")
def test_predict(

    mock_model,
    mock_collection

):

    mock_model.predict.return_value = [

        np.log1p(500)

    ]

    payload = {

        "Quantity":3,
        "Discount":0.1,
        "Profit":45,

        "Order_Month":5,
        "Order_Year":2026,
        "Order_Day":24,

        "Ship_Delay_Days":2,

        "Ship_Mode":"Second Class",
        "Segment":"Consumer",
        "Country":"United States",
        "City":"New York City",
        "State":"New York",
        "Region":"East",
        "Category":"Office Supplies",
        "Sub_Category":"Binders"

    }

    response = client.post(

        "/predict",
        json=payload

    )

    assert response.status_code == 200

    assert "Predicted Sales" in response.json()

    mock_collection.insert_one.assert_called_once()


# =========================
# ANALYTICS ENDPOINT
# =========================

@patch("backend.main.pd.read_csv")
def test_analytics(

    mock_read_csv

):

    mock_read_csv.return_value = pd.DataFrame({

        "Sales":[

            100,
            200,
            300

        ],

        "Category":[

            "Furniture",
            "Technology",
            "Technology"

        ]

    })

    response = client.get(

        "/analytics"

    )

    result = response.json()

    assert response.status_code == 200

    assert result["total_sales"] == 600

    assert result["average_sales"] == 200

    assert result["top_category"] == "Technology"


# =========================
# CHAT ENDPOINT
# =========================

@patch("backend.main.chat_collection")
@patch("backend.main.master_agent")
def test_chat(

    mock_master,
    mock_chat

):

    mock_master.return_value = (

        "Retail forecast looks good"

    )

    payload = {

        "message":

        "forecast sales"

    }

    response = client.post(

        "/chat",
        json=payload

    )

    assert response.status_code == 200

    assert (

        "response"

        in response.json()

    )

    mock_chat.insert_one.assert_called_once()


# =========================
# PREDICTION HISTORY
# =========================

@patch(
"backend.main.prediction_collection"
)
def test_prediction_history(

    mock_prediction

):

    mock_prediction.find.return_value = [

        {

            "_id":"1",

            "quantity":3,

            "discount":0.1,

            "profit":20,

            "predicted_sales":500

        }

    ]

    response = client.get(

        "/prediction-history"

    )

    assert response.status_code == 200

    assert len(

        response.json()

    ) == 1


# =========================
# CHAT HISTORY
# =========================

@patch(
"backend.main.chat_collection"
)
def test_chat_history(

    mock_chat

):

    mock_chat.find.return_value = [

        {

            "_id":"1",

            "question":"Hello",

            "response":"Hi"

        }

    ]

    response = client.get(

        "/chat-history"

    )

    assert response.status_code == 200

    assert len(

        response.json()

    ) == 1


# =========================
# FILE UPLOAD
# =========================

from unittest.mock import patch, mock_open
from io import BytesIO


@patch("backend.main.upload_to_blob")
@patch("builtins.open", new_callable=mock_open)
def test_upload(

    mock_file,
    mock_blob

):

    file = {

        "file": (

            "sample.csv",

            BytesIO(b"col1,col2\n1,2"),

            "text/csv"

        )

    }

    response = client.post(

        "/upload",

        files=file

    )

    assert response.status_code == 200

    assert response.json() == {

        "filename": "sample.csv",

        "message": "File uploaded to Azure Blob Storage"

    }

    mock_blob.assert_called_once()