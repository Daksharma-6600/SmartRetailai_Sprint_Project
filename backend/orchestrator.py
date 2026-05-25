import pandas as pd
import numpy as np
import joblib
import os

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.anomaly_detection import detect_anomalies

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

# =========================
# LOAD DATASET
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train.csv"
)

df = pd.read_csv(DATA_PATH)

# =========================
# LOAD ML MODEL
# =========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sales_forecast_model.pkl"
)

model = joblib.load(MODEL_PATH)

# LOAD MODEL COLUMNS

COLUMN_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_columns.pkl"
)

model_columns = joblib.load(
    COLUMN_PATH
)

# =========================
# AZURE OPENAI CONFIG
# =========================

llm = AzureChatOpenAI(

    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),

    api_key=os.getenv("AZURE_OPENAI_API_KEY"),

    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),

    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),

    temperature=0
)

# =========================
# LOAD EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD FAISS VECTOR DB
# =========================

FAISS_PATH = os.path.join(
    BASE_DIR,
    "rag",
    "faiss_index"
)

db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# =========================
# FORECAST AGENT
# =========================

def forecast_agent():

    sample = {

        # NUMERICAL FEATURES

        "Quantity": 3,
        "Discount": 0.0,
        "Profit": 41.91,

        "Order Month": 11,
        "Order Year": 2016,
        "Order Day": 15,

        "Ship Delay Days": 4,

        # ONE HOT ENCODED FEATURES

        "Ship Mode_Second Class": 1,
        "Segment_Consumer": 1,
        "Country_United States": 1,
        "City_Henderson": 1,
        "State_Kentucky": 1,
        "Region_South": 1,
        "Category_Furniture": 1,
        "Sub-Category_Bookcases": 1
    }

    # CREATE EMPTY DATAFRAME

    input_df = pd.DataFrame(columns=model_columns)

    input_df.loc[0] = 0

    # INSERT VALUES

    for key, value in sample.items():

        if key in input_df.columns:

            input_df.at[0, key] = value

    # MODEL PREDICTION

    prediction_log = model.predict(input_df)[0]

    # REVERSE LOG TRANSFORM

    prediction = np.expm1(prediction_log)

    return {

        "Predicted Sales":
        float(round(prediction, 2))
    }

# =========================
# ANALYTICS AGENT
# =========================

def analytics_agent():

    insights = {}

    insights["Total Sales"] = round(df["Sales"].sum(), 2)

    insights["Average Sales"] = round(df["Sales"].mean(), 2)

    insights["Top Category"] = (
        df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    insights["Top Region"] = (
        df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    insights["Top Segment"] = (
        df.groupby("Segment")["Sales"]
        .sum()
        .idxmax()
    )

    insights["Most Used Ship Mode"] = (
        df["Ship Mode"]
        .value_counts()
        .idxmax()
    )

    # DATE ANALYSIS

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    monthly_sales = (
        df.groupby(df["Order Date"].dt.month)["Sales"]
        .sum()
    )

    insights["Peak Sales Month"] = int(
        monthly_sales.idxmax()
    )

    insights["Lowest Performing Product"] = (

        df.groupby("Product Name")["Sales"]
        .sum()
        .idxmin()
    )

    return insights

def anomaly_agent():

    result = detect_anomalies()

    # print("ANOMALY RESULT:", result)

    return result

# =========================
# DOCUMENT / RAG AGENT
# =========================

def document_agent(query):

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([

        doc.page_content
        for doc in docs
    ])

    prompt = f"""

You are a Smart Retail AI Assistant.

Answer ONLY from the provided context.

CONTEXT:
{context}

QUESTION:
{query}

"""

    response = llm.invoke(prompt)

    return response.content

# =========================
# QUERY ROUTER
# =========================

def route_query(user_query):

    query = user_query.lower()

    # =====================
    # ANOMALY ROUTE
    # =====================

    if any(keyword in query for keyword in [

        "anomaly",
        "anomalies",
        "outlier",
        "outliers",
        "abnormal",
        "unusual sales",
        "anomaly detection"

    ]):

        print("ANOMALY AGENT TRIGGERED")

        result = anomaly_agent()

        prompt = f"""

You are a retail anomaly detection expert.

ANOMALY RESULTS:
{result}

Explain these anomaly findings in simple business language.

Mention:
1. Total anomalies found
2. Highest anomaly sales value
3. Business impact

Keep the answer short and professional.

"""

        return llm.invoke(prompt).content

    # =====================
    # FORECAST ROUTE
    # =====================

    elif "forecast" in query or "predict" in query:

        print("FORECAST AGENT TRIGGERED")

        result = forecast_agent()

        prompt = f"""

You are a retail forecasting expert.

Forecast Result:
{result}

Explain the prediction in simple business terms.

"""

        return llm.invoke(prompt).content

    # =====================
    # ANALYTICS ROUTE
    # =====================

    elif any(keyword in query for keyword in [

        "analytics",
        "sales",
        "insight",
        "business",
        "region",
        "category",
        "segment",
        "shipping",
        "ship",
        "shipment",
        "ship mode",
        "month",
        "product",
        "top",
        "highest",
        "lowest"

    ]):

        print("ANALYTICS AGENT TRIGGERED")

        result = analytics_agent()

        # DIRECT RESPONSE FIX

        if "ship" in query:

            return (
                f"Most Used Ship Mode: "
                f"{result['Most Used Ship Mode']}"
            )

        prompt = f"""

You are a retail data analyst.

USER QUESTION:
{user_query}

DATASET INSIGHTS:
{result}

Give a short and accurate business answer.

"""

        return llm.invoke(prompt).content

    # =====================
    # DOCUMENT / RAG ROUTE
    # =====================

    else:

        print("DOCUMENT AGENT TRIGGERED")

        return document_agent(user_query)

# =========================
# MASTER AGENT
# =========================

def master_agent(user_query):

    return route_query(user_query)