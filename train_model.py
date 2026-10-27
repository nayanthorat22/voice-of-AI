#!/usr/bin/env python
"""
train_model.py
Train a TF-IDF + Logistic Regression pipeline on the prepared Amazon Reviews dataset.
Evaluates on a held-out test split (20%) and saves metrics & pipeline artifact.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'amazon_reviews.csv')
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'sentiment_model.joblib')
METRICS_PATH = os.path.join(ARTIFACT_DIR, 'metrics.json')

def train():
    print("=" * 60)
    print("TRAINING SENTIMENT CLASSIFIER (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}. Please run prepare_data.py first.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total reviews loaded: {len(df)}")
    print(df['sentiment'].value_counts())

    # Ensure clean text
    df = df.dropna(subset=['review', 'sentiment'])
    X = df['review'].astype(str)
    y = df['sentiment'].astype(str)

    # 80/20 Stratified Split
    print("\nPerforming 80/20 stratified train/test split (seed=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Construct ML Pipeline
    # Using n-grams (1, 2), sublinear TF scaling, max_features=25000, min_df=3
    print("\nConfiguring TF-IDF vectorizer and Logistic Regression...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            min_df=3,
            sublinear_tf=True,
            strip_accents='unicode'
        )),
        ('clf', LogisticRegression(
            C=1.5,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ))
    ])

    print("Fitting model on training set...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate on held-out test set
    print("\nEvaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_p = float(precision_score(y_test, y_pred, average='macro'))
    macro_r = float(recall_score(y_test, y_pred, average='macro'))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))

    classes = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print("-" * 50)
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:  {macro_p * 100:.2f}%")
    print(f"Macro Recall:     {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1 * 100:.2f}%")
    print("-" * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    # Save artifacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"\nSaving pipeline to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)

    metrics_payload = {
        "model_name": "TF-IDF (1-2 ngrams) + Logistic Regression (C=1.5)",
        "dataset": "Amazon Reviews Polarity (Cleaned)",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "classes": classes,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "parameters": {
            "ngram_range": [1, 2],
            "max_features": 25000,
            "min_df": 3,
            "sublinear_tf": True,
            "C": 1.5,
            "solver": "lbfgs"
        }
    }

    print(f"Saving evaluation metrics to {METRICS_PATH}...")
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=4)

    print("\nSUCCESS: Model training and evaluation completed!")
    print("=" * 60)

if __name__ == '__main__':
    train()
