from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model + scaler
model = joblib.load("model/cardio_model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------------- INPUTS ----------------
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        # ---------------- FEATURES ----------------
        features = np.array([[
            age, gender, height, weight,
            ap_hi, ap_lo,
            cholesterol, gluc,
            smoke, alco, active
        ]])

        # ---------------- SCALING ----------------
        features = scaler.transform(features)

        # ---------------- PREDICTION ----------------
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "High Cardiovascular Risk ⚠️"
        else:
            result = "Low Cardiovascular Risk ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)