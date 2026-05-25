from unittest.mock import patch
from backend.orchestrator import (

    analytics_agent,
    forecast_agent,
    route_query

)

# =======================
# FORECAST AGENT
# =======================

@patch(
"backend.orchestrator.model"
)
def test_forecast(

    mock_model

):

    mock_model.predict.return_value = [

        5.5

    ]

    result = forecast_agent()

    assert (

        "Predicted Sales"

        in result

    )


# =======================
# ANALYTICS AGENT
# =======================

def test_analytics_agent():

    result = analytics_agent()

    assert (

        "Total Sales"

        in result

    )

    assert (

        "Top Category"

        in result

    )


# =======================
# ROUTER
# =======================

@patch(
"backend.orchestrator.analytics_agent"
)
def test_route_analytics(

    mock_analytics

):

    mock_analytics.return_value = {

        "Most Used Ship Mode":

        "Standard Class"

    }

    result = route_query(

        "shipping insights"

    )

    assert (

        "Standard Class"

        in result

    )