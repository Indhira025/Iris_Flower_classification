
🌸 Iris Flower Classification Web App

📌 Project Overview

This project is a Machine Learning Web Application that predicts the species of an iris flower based on its measurements.

Users can select between:

🔹 K-Nearest Neighbors (KNN)

🔹 Gaussian Naive Bayes (NB)

The app provides:

✅ Real-time predictions

✅ Model performance metrics

✅ Clean and interactive UI

🎯 Objective

Build ML models using the Iris dataset

Provide an easy-to-use prediction interface

Deploy the application online

Show accuracy and evaluation metrics

📊 Dataset Information

Dataset: Iris Flower Dataset

Total Samples: 150

Features:

    Sepal Length

    Sepal Width

    Petal Length

    Petal Width

Classes:

    Iris-setosa

    Iris-versicolor

    Iris-virginica

🛠️ Tech Stack

🔹 Backend

    Python 3.11

    Flask

    Scikit-learn

    Pandas

    NumPy

    Pickle

🔹 Frontend

    HTML5

    CSS3

🔹 Deployment & Tools

    Render

    Git & GitHub

    Gunicorn


Iris-Flower-Prediction/

    │

    ├── app.py

    ├── knn_model.pkl

    ├── nb_model.pkl

    ├── Iris.csv

    ├── requirements.txt

    ├── Procfile

    ├── templates/

    │ 
       └── index.html
 
    └── static/

       └── style.css

⚙️ Installation & Setup

1️⃣ Clone Repository

git clone https://github.com/yourusername/iris-classification-app.git
cd iris-classification-app

2️⃣ Create Virtual Environment

    python -m venv venv
    venv\Scripts\activate

3️⃣ Install Dependencies

    pip install -r requirements.txt

4️⃣ Run Application

    python app.py

Open browser:

http://127.0.0.1:5000

🤖 Models Used

🔹 K-Nearest Neighbors (KNN)

     n_neighbors = 11

     Instance-based learning

🔹 Gaussian Naive Bayes

     Probabilistic classifier

     Based on Bayes theorem

📈 Model Performance
| Model | Dataset | Accuracy |
| ----- | ------- | -------- |
| KNN   | Train   | 0.9583   |
| KNN   | Test    | 1.0      |
| NB    | Train   | 0.95     |
| NB    | Test    | 1.0      |


✔ Both models achieved excellent performance.

🚀 Features

🔸 Real-time prediction

🔸 Model selection (KNN / NB)

🔸 Accuracy display

🔸 Confusion matrix

🔸 Classification report

🔸 Clean UI

🌐 Deployment (Render)

Steps:

    Push code to GitHub

    Create Web Service on Render

Configure:

    Build: pip install -r requirements.txt

    Start: gunicorn app:app

Deploy

📌 Sample Input

    Sepal Length: 5.1

    Sepal Width: 3.5

    Petal Length: 1.4

    Petal Width: 0.2
    
Model: KNN

Output:

Prediction: Iris-setosa

Train Accuracy: 0.9583

Test Accuracy: 1.0

⚠️ Challenges & Solutions
| Challenge         | Solution                  |
| ----------------- | ------------------------- |
| Model saving      | Used Pickle               |
| Input format      | Converted to DataFrame    |
| Metrics display   | Computed dynamically      |
| Deployment errors | Added Gunicorn            |
| JSON issues       | Converted arrays to lists |

🔮 Future Enhancements

    Add more ML models (SVM, Random Forest)

    Improve UI (mobile-friendly)

    Add database for history

    Visualizations

    Docker deployment

✅ Conclusion

This project demonstrates a complete end-to-end ML pipeline:

Data preprocessing

Model training

Web integration

Deployment

It’s a great beginner-friendly ML + Web project.

📚 References

Flask Documentation

Scikit-learn

Render Docs

Kaggle Iris Dataset

Contact:

Name : Rongali Indhira

Email : indhirarongali123@gmail.com

PhoneNumber : 8096488064

LinkedIn : https://www.linkedin.com/in/rongali-indhira-6599312b4/
