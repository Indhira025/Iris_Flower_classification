<h1 align="center">🌸 Iris Flower Classification Web Application</h1>

<p align="center">
A Machine Learning Web Application built using <b>Python, Flask, and Scikit-learn</b>  
<br>
Deployed on <b>Render</b> with Gunicorn
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This project predicts the species of an iris flower using four input features:
sepal length, sepal width, petal length, and petal width.
</p>

<ul>
<li>✔ Real-time species prediction</li>
<li>✔ Two ML models (KNN & Naive Bayes)</li>
<li>✔ User-friendly web interface</li>
<li>✔ Displays model performance metrics</li>
</ul>

<hr>

<h2>📊 Dataset Information</h2>

<ul>
<li><b>Dataset:</b> Iris Flower Dataset</li>
<li><b>Source:</b> Kaggle</li>
<li><b>Total Samples:</b> 150</li>
<li><b>Features:</b> 4</li>
<li><b>Target Classes:</b> 3 (Setosa, Versicolor, Virginica)</li>
</ul>

<hr>

<h2>🧠 Model Development</h2>

<p><b>Algorithms Used:</b></p>
<ul>
<li>K-Nearest Neighbors (KNN)</li>
<li>Gaussian Naive Bayes</li>
</ul>

<p><b>Evaluation Metrics:</b></p>
<ul>
<li>Accuracy</li>
<li>Confusion Matrix</li>
<li>Classification Report</li>
</ul>

<hr>

<h2>💻 Technology Stack</h2>

<h3>Backend</h3>
<ul>
<li>Python 3.11</li>
<li>Flask</li>
<li>Scikit-learn</li>
<li>Pandas</li>
<li>NumPy</li>
<li>Pickle</li>
</ul>

<h3>Frontend</h3>
<ul>
<li>HTML5</li>
<li>CSS3</li>
</ul>

<h3>Deployment</h3>
<ul>
<li>GitHub</li>
<li>Gunicorn</li>
<li>Render</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
Iris-Flower-Prediction/
│
├── app.py
├── knn_model.pkl
├── nb_model.pkl
├── Iris.csv
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
├── static/
└── README.md
</pre>

<hr>

<h2>🚀 Installation & Setup</h2>

<pre>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
</pre>

<p>Open in browser:</p>

<pre>
http://127.0.0.1:5000/
</pre>

<hr>

<h2>🌍 Live Demo</h2>

<p>
🔗 <a href="#" target="_blank">
Live Application
</a>
</p>

<p>
💻 <a href="#" target="_blank">
GitHub Repository
</a>
</p>

<hr>

<h2>📈 Sample Prediction</h2>

<ul>
<li>Sepal Length: 5.1</li>
<li>Sepal Width: 3.5</li>
<li>Petal Length: 1.4</li>
<li>Petal Width: 0.2</li>
<li>Model: KNN</li>
</ul>

<p><b>Predicted Output:</b> Iris-setosa</p>

<hr>

<h2>📊 Model Performance</h2>

<ul>
<li>KNN Train Accuracy: 0.9583</li>
<li>KNN Test Accuracy: 1.0</li>
<li>Naive Bayes Train Accuracy: 0.95</li>
<li>Naive Bayes Test Accuracy: 1.0</li>
</ul>

<hr>

<h2>🔮 Future Enhancements</h2>

<ul>
<li>Random Forest / SVM implementation</li>
<li>Visualization of decision boundaries</li>
<li>Prediction history storage</li>
<li>Mobile-friendly UI</li>
<li>Docker deployment</li>
</ul>

<hr>

<h2>👩‍💻 Author</h2>

<p>
<b>R. Indhira</b><br>
Machine Learning & Python Developer<br>
Last Updated: 16/02/2026
</p>
