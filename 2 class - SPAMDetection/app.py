from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the trained TF-IDF + Naive Bayes pipeline
model = load("naive_bayes_message_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get message entered by the user
    message = request.form.get("message", "").strip()

    # Check whether the user entered anything
    if not message:
        return render_template(
            "index.html",
            error="Please enter a message."
        )

    # Make prediction
    prediction = model.predict([message])[0]

    # Get prediction probabilities
    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)

    # Format confidence as percentage
    confidence_percent = confidence * 100

    return render_template(
        "index.html",
        message=message,
        prediction=prediction,
        confidence=f"{confidence_percent:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)