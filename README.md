# 🌿 ML Classifier with Flask API

This project implements a **Decision Tree Classifier** using the **Iris Dataset** and exposes it via a Flask API.  
The API supports:  
- 🚀 **`/predict`** → Predict the class of an Iris flower  
- 📊 **`/get-status`** → Get the train-test split counts  

---

## 📌 1️⃣ Setting up the Environment
To isolate dependencies, set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # For macOS
```

## 📌 2️⃣ Install Dependencies
To isolate dependencies, set up a virtual environment:

```bash
pip install -r requirements.txt
```

## 📌 3️⃣ Train the Model
Before running the API, train the classifier using the following command:

```bash
python train.py
```


## 📌 4️⃣ Run the Flask API
start the Flask server:

```bash
python app.py
```


## 📌 5️⃣ API Endpoints
🚀 POST /predict - Make a classification

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

```

```json
{
  "prediction": "Iris-setosa"
}

```

## 📊 GET /get-status - Get train-test split details

```bash
curl -X GET http://127.0.0.1:5000/get-status
```

```json 

{
  "train_count": 120,
  "test_count": 30
}

```