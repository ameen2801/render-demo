import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get all form values as floats
    float_features = [float(x) for x in request.form.values()]

    # Create numpy array and reshape for prediction
    features = [np.array(float_features)]

    # Make prediction
    prediction = model.predict(features)

    # Map prediction to meaningful output
    group_name = "Group 1" if prediction[0] == 1 else "Group 2"

    return render_template("index.html", prediction_text=f"The patient belongs to {group_name}")


if __name__ == "__main__":
        app.run()