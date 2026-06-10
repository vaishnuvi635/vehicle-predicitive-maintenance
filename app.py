from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("random_forest_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["Mileage"]),
        float(request.form["Maintenance_History"]),
        float(request.form["Reported_Issues"]),
        float(request.form["Vehicle_Age"]),
        float(request.form["Engine_Size"]),
        float(request.form["Odometer_Reading"]),
        float(request.form["Warranty_Expiry_Date"]),
        float(request.form["Insurance_Premium"]),
        float(request.form["Service_History"]),
        float(request.form["Accident_History"]),
        float(request.form["Fuel_Efficiency"]),
        float(request.form["Tire_Condition"]),
        float(request.form["Brake_Condition"]),
        float(request.form["Battery_Status"]),
        float(request.form["Days_Since_Last_Service"])
    ]

    data = np.array([features])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    health_score = round((1 - probability) * 100, 2)

    if prediction == 1:
        result = "Maintenance Required"
        status = "danger"
    else:
        result = "No Maintenance Required"
        status = "success"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2),
        health_score=health_score,
        status=status
    )

if __name__ == "__main__":
    app.run(debug=True)