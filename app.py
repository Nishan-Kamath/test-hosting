from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

# Load saved model, scaler, and encoder
model = joblib.load("gender_model_3features.pkl")
scaler = joblib.load("scaler_3features.pkl")
encoder = joblib.load("label_encoder_3features.pkl")


app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session

@app.route("/", methods=["GET"])
def home():
    # Get prediction from session if available
    prediction_text = session.pop("prediction_text", None)
    return render_template("index.html", prediction_text=prediction_text)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs
        meanfreq = float(request.form["meanfreq"])
        sd = float(request.form["sd"])
        median = float(request.form["median"])

        # Prepare features (pad with zeros)
        features = np.array([[meanfreq, sd, median] + [0]*(model.n_features_in_ - 3)])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        gender = encoder.inverse_transform([prediction])[0]

        # Store prediction in session and redirect to "/"
        session["prediction_text"] = f"Prediction: {gender}"
        return redirect(url_for("home"))

    except Exception as e:
        session["prediction_text"] = f"Error: {str(e)}"
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
