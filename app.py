from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("random_forest_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form

    mileage = float(request.form["Mileage"])
    maintenance_history = float(request.form["Maintenance_History"])
    reported_issues = float(request.form["Reported_Issues"])
    vehicle_age = float(request.form["Vehicle_Age"])
    engine_size = float(request.form["Engine_Size"])
    odometer_reading = float(request.form["Odometer_Reading"])
    warranty_expiry = float(request.form["Warranty_Expiry_Date"])
    insurance_premium = float(request.form["Insurance_Premium"])
    service_history = float(request.form["Service_History"])
    accident_history = float(request.form["Accident_History"])

    tire_condition = float(request.form["Tire_Condition"])
    brake_condition = float(request.form["Brake_Condition"])
    battery_status = float(request.form["Battery_Status"])

    days_since_service = float(
        request.form["Days_Since_Last_Service"]
    )

    # Fuel Efficiency removed from UI
    # Keep a default value for model compatibility

    fuel_efficiency = 15.0

    # EXACT feature order used during training

    features = [
        mileage,
        maintenance_history,
        reported_issues,
        vehicle_age,
        engine_size,
        odometer_reading,
        warranty_expiry,
        insurance_premium,
        service_history,
        accident_history,
        fuel_efficiency,
        tire_condition,
        brake_condition,
        battery_status,
        days_since_service
    ]

    data = np.array([features])

    # Scale input

    scaled_data = scaler.transform(data)

    # Prediction

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    probability_percent = round(probability * 100, 2)

    health_score = round((1 - probability) * 100, 2)

    # Smart Recommendations

    recommendations = []

    if prediction == 1:

        if tire_condition == 0:
            recommendations.append(
                "Inspect or replace worn tires."
            )

        if brake_condition == 0:
            recommendations.append(
                "Inspect brake pads and braking system."
            )

        if battery_status == 0:
            recommendations.append(
                "Check battery condition and charging system."
            )

        if reported_issues > 2:
            recommendations.append(
                "Review and resolve reported vehicle issues."
            )

        if days_since_service > 180:
            recommendations.append(
                "Schedule routine servicing immediately."
            )

        if accident_history > 1:
            recommendations.append(
                "Perform a complete vehicle inspection after accident history."
            )

        if len(recommendations) == 0:
            recommendations.append(
                "General maintenance is recommended based on overall vehicle condition."
            )

        result = "Maintenance Required"
        status = "danger"

    else:

        recommendations = [
            "Vehicle is operating normally.",
            "Continue regular maintenance schedule.",
            "Perform routine inspections as recommended."
        ]

        result = "No Maintenance Required"
        status = "success"

    return render_template(
        "index.html",
        prediction=result,
        probability=probability_percent,
        health_score=health_score,
        status=status,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
