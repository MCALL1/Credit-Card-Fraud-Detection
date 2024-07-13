# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:36:55 2024

@author: mcall
"""

# Credit Card Fraud Detection

## Description
#This project aims to develop a machine learning model to detect fraudulent credit card transactions. The dataset used is from Kaggle.

## Dataset
# [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Project Structure
# Data: Raw and preprocessed data.
# Notebooks: EDA, model building, and evaluation.
# Scripts: Data preprocessing, modeling, and evaluation scripts.
# Models: Saved models.
# Visualizations: Plots and charts.
# Deployment: Code for deploying the model using Flask.

## How to Run
#1. Clone the repository.
#2. Install required packages.
#3. Run the notebooks/scripts.

## Results
#Summary of findings and model performance.
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('data/creditcard.csv')

# Check for missing values
print(data.isnull().sum())

# Split data into features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Save processed data
X_train_smote.to_csv('data/X_train_smote.csv', index=False)
y_train_smote.to_csv('data/y_train_smote.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load processed data
X_train = pd.read_csv('data/X_train_smote.csv')
y_train = pd.read_csv('data/y_train_smote.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train.values.ravel())

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_pred))

# Save model
joblib.dump(model, 'models/random_forest.pkl')
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('models/random_forest.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
