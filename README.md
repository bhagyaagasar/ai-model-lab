# AI Model Lab 🤖

A portfolio-ready machine-learning web application for dataset exploration, model training, evaluation, and prediction.

## Features

- CSV upload
- Dataset profiling
- Missing-value inspection
- Automatic categorical encoding
- Train/test split
- Four classification algorithms:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine
  - K-Nearest Neighbors
- Accuracy, precision, recall and F1-score
- Confusion matrix
- Classification report
- Single-row prediction
- Prediction probabilities when supported

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Suggested GitHub description

> Interactive AI/ML laboratory for dataset exploration, model training, evaluation, confusion-matrix visualization, and real-time classification using Scikit-learn and Streamlit.

## Future extensions

- Regression models
- Hyperparameter tuning
- ROC and Precision-Recall curves
- Feature importance / SHAP explanations
- Model download
- Experiment history
- Image classification with CNNs
- YOLO object detection
- Authentication and project workspaces
