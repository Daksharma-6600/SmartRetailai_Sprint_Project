import pandas as pd
import numpy as np
import joblib



df = pd.read_csv("../dataset/train.csv")

# LOAD MODEL

model = joblib.load("../models/sales_forecast_model.pkl")


# LOAD TRAINING COLUMNS


model_columns = joblib.load("../models/model_columns.pkl")


# FORECAST AGENT


def forecast_agent():

    sample = {

        # Numerical Features

        "Quantity": 3,
        "Discount": 0.0,
        "Profit": 41.91,
        "Order Month": 11,
        "Order Year": 2016,
        "Order Day": 15,
        "Ship Delay Days": 4,

        # One Hot Encoded Features

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

    # FILL ALL COLUMNS WITH 0

    input_df.loc[0] = 0

    # INSERT SAMPLE VALUES

    for key, value in sample.items():

        if key in input_df.columns:

            input_df.at[0, key] = value

    # PREDICTION (LOG SCALE)

    prediction_log = model.predict(input_df)[0]

    # REVERSE LOG TRANSFORM

    prediction = np.expm1(prediction_log)

    return {

        "Predicted Sales":
        float(round(prediction, 2))
    }



def analytics_agent(query=None):

    insights = {}

    
    # TOTAL SALES
    

    insights["Total Sales"] = round(
        df["Sales"].sum(),
        2
    )

   
    # AVERAGE SALES
   

    insights["Average Sales"] = round(
        df["Sales"].mean(),
        2
    )

    
    # TOP CATEGORY
    

    insights["Top Category"] = (
        df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

 
    # TOP REGION
   

    insights["Top Region"] = (
        df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

   
    # TOP CUSTOMER SEGMENT
    

    insights["Top Segment"] = (
        df.groupby("Segment")["Sales"]
        .sum()
        .idxmax()
    )

    
    # MOST USED SHIPPING MODE
  

    insights["Most Used Ship Mode"] = (
        df["Ship Mode"]
        .value_counts()
        .idxmax()
    )

 
    # PEAK SALES MONTH
    

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    monthly_sales = (
        df.groupby(
            df["Order Date"].dt.month
        )["Sales"]
        .sum()
    )

    peak_month = monthly_sales.idxmax()

    insights["Peak Sales Month"] = int(
        peak_month
    )

    
    # UNDERPERFORMING PRODUCT
    
    low_product = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .idxmin()
    )

    insights["Lowest Performing Product"] = low_product

    return insights


# DOCUMENT AGENT


def document_agent():

    return {

        "Status":
        "RAG Chatbot Ready"
    }


# TEST AGENTS


print("\nForecast Agent\n")
print(forecast_agent())

print("\nAnalytics Agent\n")
print(analytics_agent())

print("\nDocument Agent\n")
print(document_agent())