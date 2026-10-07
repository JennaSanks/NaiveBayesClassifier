from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load trained model
model = load("college_message_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form.get("message", "").strip()

    if not message:
        return render_template(
            "index.html",
            error="Please enter a message."
        )

    # Predict class
    prediction = model.predict([message])[0]

    # Get probabilities
    probabilities = model.predict_proba([message])[0]

    confidence = max(probabilities) * 100

    return render_template(
        "index.html",
        message=message,
        prediction=prediction,
        confidence=f"{confidence:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)