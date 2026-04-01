from fastapi import FastAPI
from fastapi.responses import HTMLResponse   
import joblib

app = FastAPI()

# Load model
model = joblib.load("risk_model.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1> EarnKavach Risk API</h1>
    <p>API is running successfully!</p>
    <p> Go to <a href="/docs">/docs</a> to test API</p>
    """


# 👇 KEEP YOUR EXISTING API BELOW
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
