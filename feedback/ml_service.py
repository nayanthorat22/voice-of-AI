"""
feedback/ml_service.py
Loads the trained ML Pipeline (TF-IDF + Logistic Regression) from artifacts/sentiment_model.joblib.
Provides singleton caching, prediction, probability calculation, and helpful fallbacks.
"""

import os
import joblib
import numpy as np

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'sentiment_model.joblib')
METRICS_PATH = os.path.join(ARTIFACT_DIR, 'metrics.json')

class ModelNotLoadedError(Exception):
    pass

class SentimentMLService:
    _instance = None
    _pipeline = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.load_model()

    def load_model(self):
        """Loads or reloads the trained model from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                self._pipeline = joblib.load(MODEL_PATH)
            except Exception as e:
                self._pipeline = None
                print(f"Warning: Failed to load sentiment model: {e}")
        else:
            self._pipeline = None

    @property
    def is_available(self):
        return self._pipeline is not None

    def predict_one(self, text: str):
        """
        Classifies a single review text string.
        Returns:
            dict: {
                'sentiment': 'Positive' | 'Negative',
                'confidence': float (0.0 to 1.0),
                'probabilities': {'Positive': float, 'Negative': float},
                'is_low_confidence': bool
            }
        """
        if not self.is_available:
            self.load_model()
            if not self.is_available:
                raise ModelNotLoadedError(
                    "Trained ML model not found. Please run 'python train_model.py' to generate artifacts/sentiment_model.joblib."
                )

        clean_text = str(text).strip()
        if not clean_text:
            return {
                'sentiment': 'Neutral',
                'confidence': 0.50,
                'probabilities': {'Positive': 0.5, 'Negative': 0.5},
                'is_low_confidence': True
            }

        # Predict with sklearn pipeline
        pred_label = self._pipeline.predict([clean_text])[0]

        # Extract probability if available
        if hasattr(self._pipeline, 'predict_proba'):
            probs = self._pipeline.predict_proba([clean_text])[0]
            classes = list(self._pipeline.classes_)
            prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
            confidence = prob_dict.get(pred_label, float(np.max(probs)))
        else:
            confidence = 0.85
            prob_dict = {pred_label: 0.85}

        confidence = round(confidence, 4)
        is_low = confidence < 0.65

        return {
            'sentiment': pred_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'is_low_confidence': is_low
        }

    def predict_batch(self, texts: list):
        """
        Efficient vectorised batch prediction for CSV uploads.
        Returns:
            list of dicts containing 'sentiment' and 'confidence'
        """
        if not self.is_available:
            self.load_model()
            if not self.is_available:
                raise ModelNotLoadedError(
                    "Trained ML model not found. Please run 'python train_model.py' before uploading reviews."
                )

        clean_texts = [str(t) if t is not None else "" for t in texts]
        preds = self._pipeline.predict(clean_texts)

        results = []
        if hasattr(self._pipeline, 'predict_proba'):
            probs_matrix = self._pipeline.predict_proba(clean_texts)
            classes = list(self._pipeline.classes_)
            
            for i, pred in enumerate(preds):
                row_probs = probs_matrix[i]
                prob_dict = {c: float(row_probs[idx]) for idx, c in enumerate(classes)}
                conf = prob_dict.get(pred, float(np.max(row_probs)))
                results.append({
                    'sentiment': pred,
                    'confidence': round(conf, 4)
                })
        else:
            for pred in preds:
                results.append({
                    'sentiment': pred,
                    'confidence': 0.85
                })

        return results

# Singleton access function
def get_ml_service() -> SentimentMLService:
    return SentimentMLService.get_instance()
