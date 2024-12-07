# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn

app = FastAPI(title="Accident Prediction API")

class PredictionRequest(BaseModel):
    year: int
    month: int

with open('auto_arima_model.pkl', 'rb') as file:
    model = pickle.load(file)


LAST_TRAINING_DATE = pd.to_datetime('2020-12-01')

@app.post("/predict", response_model=dict)
def predict(request: PredictionRequest):
    """
    Predict the number of 'Alkoholunfälle' (alcohol-related accidents) for a given year and month.
    """
    input_year = request.year
    input_month = request.month

    # Validate the input month
    if not 1 <= input_month <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")

    # Create a datetime object for the requested date
    try:
        requested_date = pd.to_datetime(f"{input_year}-{input_month:02d}-01")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year or month.")

    # Check if the requested date is after the last training date
    if requested_date <= LAST_TRAINING_DATE:
        raise HTTPException(status_code=400, detail="Requested date must be after the last training date (2020-12).")

    # Calculate the number of months ahead to forecast
    months_ahead = (requested_date.year - LAST_TRAINING_DATE.year) * 12 + (requested_date.month - LAST_TRAINING_DATE.month)

    # Limit the forecast horizon to prevent excessive computation
    MAX_FORECAST_HORIZON = 12  # e.g., up to 12 months ahead
    if months_ahead > MAX_FORECAST_HORIZON:
        raise HTTPException(status_code=400, detail=f"Please request a forecast within {MAX_FORECAST_HORIZON} months ahead.")

    # Make the forecast
    try:
        forecast, conf_int = model.predict(n_periods=months_ahead, return_conf_int=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    predicted_value = forecast[-1]

    lower_ci = conf_int[-1, 0]
    upper_ci = conf_int[-1, 1]

    response = {
        "prediction": round(float(predicted_value), 2)
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
