from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load models
knn_model = pickle.load(open("knn_model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))

# Load dataset
df = pd.read_csv("Iris.csv")
df = df.drop(['Id'], axis=1)

df['Species'] = df['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class_names = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    sl = float(data["sl"])
    sw = float(data["sw"])
    pl = float(data["pl"])
    pw = float(data["pw"])
    model_type = data["model"]

    input_df = pd.DataFrame(
        [[sl, sw, pl, pw]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    model = knn_model if model_type == "knn" else nb_model

    prediction = model.predict(input_df)[0]

    # Train Performance
    y_train_pred = model.predict(X_train)
    train_accuracy = round(accuracy_score(y_train, y_train_pred), 4)
    train_cm = confusion_matrix(y_train, y_train_pred).tolist()
    train_report = classification_report(
        y_train, y_train_pred, output_dict=True
    )

    # Test Performance
    y_test_pred = model.predict(X_test)
    test_accuracy = round(accuracy_score(y_test, y_test_pred), 4)
    test_cm = confusion_matrix(y_test, y_test_pred).tolist()
    test_report = classification_report(
        y_test, y_test_pred, output_dict=True
    )

    return jsonify({
        "prediction_number": int(prediction),
        "prediction_name": class_names[prediction],
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_cm": train_cm,
        "test_cm": test_cm,
        "train_report": train_report,
        "test_report": test_report
    })

if __name__ == "__main__":
    app.run(debug=True)
