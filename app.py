from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import io
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Manufacturing Predictor API")

# Global variables to store model and scaler
model = None
scaler = None
required_columns = ['Machine_ID', 'Temperature',
                    'Run_Time']  # Machine_ID still required in data but not used for prediction
feature_columns = ['Temperature', 'Run_Time']  # Only these features will be used for prediction


class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Manufacturing Predictor API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Manufacturing Predictor API</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><code>POST /upload</code> - Upload training data (CSV file)</li>
                <li><code>POST /train</code> - Train the model</li>
                <li><code>POST /predict</code> - Make predictions</li>
            </ul>
            <p>For interactive API documentation, visit: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """


def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Machine_ID': np.random.randint(1, 11, n_samples),
        'Temperature': np.random.normal(75, 15, n_samples),
        'Run_Time': np.random.normal(100, 30, n_samples),
    }

    # Generate Downtime_Flag based on simple rules (only using Temperature and Run_Time)
    data['Downtime_Flag'] = (
            (data['Temperature'] > 90) |
            (data['Run_Time'] > 150) |
            (np.random.random(n_samples) < 0.1)
    ).astype(int)

    return pd.DataFrame(data)


def clean_numeric_data(df):
    """Clean and convert numeric columns to float."""
    for col in ['Temperature', 'Run_Time']:
        # Remove any leading/trailing whitespace
        df[col] = df[col].astype(str).str.strip()
        # Remove any non-numeric characters (except decimal points)
        df[col] = df[col].str.replace(r'[^\d.-]', '', regex=True)
        # Convert to float, replacing errors with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert Machine_ID to integer (for data organization only)
    df['Machine_ID'] = pd.to_numeric(df['Machine_ID'], errors='coerce').astype('Int64')

    # Drop rows with NaN values
    df = df.dropna()

    return df


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate required columns
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="Missing required columns")

        # Clean numeric data
        df = clean_numeric_data(df)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No valid data after cleaning")

        # Save uploaded data
        df.to_csv("uploaded_data.csv", index=False)
        return {"message": "Data uploaded successfully", "shape": df.shape}

    except Exception as e:
        logger.error(f"Error in upload_data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
async def train_model():
    try:
        # Try to load uploaded data, if not available use synthetic data
        try:
            df = pd.read_csv("uploaded_data.csv")
            logger.info("Loaded uploaded data")
        except FileNotFoundError:
            df = generate_sample_data()
            df.to_csv("uploaded_data.csv", index=False)
            logger.info("Generated synthetic data")

        # Clean data
        df = clean_numeric_data(df)
        logger.info(f"Data shape after cleaning: {df.shape}")

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No valid data available for training")

        # Prepare features and target - only using Temperature and Run_Time
        X = df[feature_columns]  # Only using relevant features
        y = df['Downtime_Flag']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        global scaler, model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))
        }

        # Save model and scaler
        joblib.dump(model, "model.joblib")
        joblib.dump(scaler, "scaler.joblib")

        return metrics

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        global model, scaler

        # Load model and scaler if not in memory
        if model is None:
            try:
                model = joblib.load("model.joblib")
                scaler = joblib.load("scaler.joblib")
            except FileNotFoundError:
                raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

        # Prepare input data - only using Temperature and Run_Time
        input_df = pd.DataFrame([{
            'Temperature': input_data.Temperature,
            'Run_Time': input_data.Run_Time
        }])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = bool(model.predict(input_scaled)[0])
        confidence = float(max(model.predict_proba(input_scaled)[0]))

        return {
            "Downtime": "Yes" if prediction else "No",
            "Confidence": confidence
        }

    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)