# Iris Flower Classification API

This project is a FastAPI application that serves a K-Nearest Neighbors (KNN) machine learning model for **Iris flower classification**.  
It predicts the species of an iris flower (`setosa`, `versicolor`, `virginica`) based on 4 numerical features.

## Features

- Predict iris species using `/predict` endpoint
- Returns prediction along with confidence score
- Health check endpoint at `/`
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
