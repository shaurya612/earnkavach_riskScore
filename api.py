from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import joblib

app = FastAPI()

model = joblib.load("risk_model.pkl")

# 👇 Redirect to docs
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")


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
