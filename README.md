# 🦺 Industrial Safety Incident Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Active-green)

## About My Contribution
This repository is based on a forked dataset structure but the **entire solution — 
data pipeline, model experimentation, Flask API, and frontend — was designed, 
developed, and implemented end-to-end by myself, Soumita Chowdhury**.

---

## Problem Statement

Industrial accidents cost lives and billions in losses every year. When an incident 
occurs, safety professionals need to quickly assess its severity to prioritise 
response. This project builds a machine learning system that classifies industrial 
safety incidents by severity level (I–V) based on a structured incident report, 
helping organisations respond faster and more accurately.

---

## Solution Overview

A multi-class text and feature classification pipeline that:
- Accepts structured incident report data as input
- Preprocesses and augments the dataset to handle class imbalance
- Evaluates 10 ML models + a deep learning model (Keras LSTM)
- Serves predictions via a Flask REST API
- Exposes reusable methods for upload, cleaning, augmentation, and model loading

---

## Dataset

- **Source:** [Industrial Safety and Health Analytics Database — Kaggle](https://www.kaggle.com/ihmstefanini/industrial-safety-and-health-analytics-database)
- **Subset used:** 425 rows (representative sample for model development)
- **Target variable:** Accident severity level (5 classes: I to V)

---

## Model Performance

### Machine Learning Models

| Model | Test Accuracy | F1-Score |
|---|---|---|
| CatBoostClassifier | **95.57%** | **0.9557** |
| RandomForestClassifier | 94.49% | 0.9453 |
| SVC | 93.52% | 0.9359 |
| XGBClassifier | 93.74% | 0.9381 |
| LogisticRegression | 93.09% | 0.9308 |
| GradientBoostingClassifier | 88.98% | 0.8911 |
| DecisionTreeClassifier | 90.71% | 0.9075 |
| KNeighborsClassifier | 92.55% | 0.9257 |
| BaggingClassifier | 94.17% | 0.9417 |
| AdaBoostClassifier | 45.79% | 0.4553 |

### Deep Learning Models (Keras)

| Model | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|---|---|---|---|---|
| Simple Neural Network | 1.377 | 38.83% | 1.583 | 34.67% |
| LSTM | 1.609 | 20.02% | 1.609 | 20.09% |
| Bidirectional LSTM | **0.039** | **99.14%** | **0.155** | **94.60%** |

> **Best deep learning model: Bidirectional LSTM** (94.60% test accuracy)  
> Simple NN and LSTM underperformed — likely due to limited data (425 rows) 
> and insufficient sequence structure for standard LSTM to learn from.

---

## Tech Stack

- **Language:** Python 3.8+
- **ML:** scikit-learn, CatBoost, XGBoost, Keras/TensorFlow
- **API:** Flask
- **Frontend:** Bootstrap + jQuery *(Work in Progress — UI integration 
  with Flask API under active debugging)*
- **Data:** pandas, NumPy, NLTK
---

```
## Project Structure

Industrial-Safety-chatbot/
├── app.py                  # Flask API — upload, clean, augment, predict
├── src/                    # Core ML pipeline modules
├── data/                   # Dataset files
├── models/                 # Saved model files (.pkl, .h5)
├── requirements.txt        # Dependencies
└── README.md

```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/soumita20/Industrial-Safety-chatbot.git
cd Industrial-Safety-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Flask API
python app.py

# 4. API will be available at http://localhost:5000
```

---

## Key Learnings

- CatBoost outperformed all other models on this tabular dataset, 
  including XGBoost and Random Forest
- Deep learning (Keras) achieved comparable accuracy but showed signs 
  of overfitting (train: 99.1% vs test: 94.6%) — addressable with 
  more data or regularisation
- Data augmentation was critical given the small subset (425 rows)
- AdaBoost performed poorly on this dataset — likely due to sensitivity 
  to the class imbalance

---

## Author

**Soumita Chowdhury**   
[GitHub](https://github.com/soumita20) | 
[LinkedIn](https://www.linkedin.com/in/soumita-chowdhury-93934617/)

---

## Roadmap

- [ ] Fix frontend–API integration (jQuery → Flask)
- [ ] Add Docker support for containerised deployment
- [ ] Expand dataset beyond 425 rows
- [ ] Add SHAP explainability for model predictions

