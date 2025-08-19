# Iris Flower Classification API

This project is a FastAPI application that serves a K-Nearest Neighbors (KNN) machine learning model for **Iris flower classification**.
It predicts the species of an iris flower (`setosa`, `versicolor`, `virginica`) based on 4 numerical features.

## Features

- Predict iris species using `/predict` endpoint
- Returns prediction along with confidence score
- Health check endpoint at `/` or `/plant-check`
- Model metadata endpoint at `/model-info`
- Input validation using Pydantic
- JSON response format

## Dataset

The model is trained on the **Iris dataset** with the following features:

- `sepal_length` (float)
- `sepal_width` (float)
- `petal_length` (float)
- `petal_width` (float)

Target classes:

| Class      | Numeric Label |
| ---------- | ------------- |
| setosa     | 0             |
| versicolor | 1             |
| virginica  | 2             |

## Model

- Algorithm: K-Nearest Neighbors (KNN)
- Saved as `model.pkl` using `joblib`
- Predicts species and provides confidence score

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

- Health check: `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/plant-check`
- Swagger docs: `http://127.0.0.1:8000/docs`

---

## API Usage

### 1. plant_check

**GET /** or **GET /plant-check**
Response example:

```json
{
  "status": "Flower_Type",
  "message": "API and model are running"
}
```

### 2. Predict Iris Species

**POST /predict**
Request example:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response example:

```json
{
  "prediction": "setosa",
  "confidence": 1.0
}
```

### 3. Model Info

**GET /model-info**
Response example:

```json
{
  "model_type": "KNN",
  "problem_type": "classification",
  "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
}
```

## Notes

- Ensure `model.pkl` is in the same folder as `main.py`
- Input values must be floats
- Confidence is the highest probability among predicted classes
