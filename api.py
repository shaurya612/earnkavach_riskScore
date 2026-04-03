from fastapi import FastAPI
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("risk_model.pkl")

@app.get("/")
def home():
    return {"message": "Risk API is running"}

@app.post("/predict-risk")
def predict_risk(data: dict):
    values = [[
        data['rain'],
        data['aqi'],
        data['traffic'],
        data['zone_risk'],
        data['disruptions']
    ]]
    
    prediction = model.predict(values)
    
    return {"risk_score": float(prediction[0])}
