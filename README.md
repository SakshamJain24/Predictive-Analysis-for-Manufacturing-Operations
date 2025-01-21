# Manufacturing Predictive Analysis API

A FastAPI-based REST API that predicts machine downtime based on operational parameters (Temperature and Run Time). The system uses a Random Forest classifier to provide predictions with confidence scores.

## Features
- Upload custom manufacturing datasets (CSV format)
- Train machine learning model on uploaded data
- Make downtime predictions with confidence scores
- Auto-generated Swagger documentation
- Built-in data validation and error handling

## Requirements
```
fastapi>=0.104.0
uvicorn>=0.24.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
python-multipart>=0.0.6
joblib>=1.3.0
pydantic>=2.4.0
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd manufacturing-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
uvicorn app:app --reload
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Upload Dataset
```bash
POST /upload
```
- Upload your manufacturing dataset (CSV)
- Required columns: Temperature, Run_Time, Downtime_Flag
- Example using curl:
```bash
curl -X POST http://localhost:8000/upload -F "file=@Synthetic_Manufacturing_Dataset.csv"
```

#### 2. Train Model
```bash
POST /train
```
- Trains the model on uploaded dataset
- Returns accuracy and F1-score metrics
- Example using curl:
```bash
curl -X POST http://localhost:8000/train
```
- Example response:
```json
{
    "accuracy": 0.87,
    "f1_score": 0.85
}
```

#### 3. Make Prediction
```bash
POST /predict
```
- Input: Temperature and Run_Time values
- Returns: Downtime prediction with confidence score
- Example using curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Temperature": 85.5, "Run_Time": 120.3}'
```
- Example response:
```json
{
    "Downtime": "Yes",
    "Confidence": 0.89
}
```

## Data Format

The API expects CSV files with the following columns:
- `Temperature`: Float (operational temperature)
- `Run_Time`: Float (machine runtime in minutes)
- `Downtime_Flag`: Integer (0 for no downtime, 1 for downtime)

Example data format:
```csv
Temperature,Run_Time,Downtime_Flag
75.5,120.3,0
92.1,145.7,1
68.4,98.2,0
```

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Error Handling

The API includes comprehensive error handling for:
- Missing or invalid columns in uploaded data
- Invalid data formats
- Model not trained before prediction
- Server errors during processing

## Notes
- The model uses Temperature and Run_Time as features to predict machine downtime
- Machine_ID is not used in predictions as it's just an identifier
- The system uses Random Forest Classifier for better accuracy and confidence scores
- All numeric data is automatically cleaned and validated during upload
