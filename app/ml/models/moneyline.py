"""Moneyline model - predicts P(home wins) using GradientBoostingClassifier."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

from app.core.constants import FEATURE_NAMES


class MoneylineModel:
    """
    Classification model that predicts probability of home team winning.

    Uses GradientBoostingClassifier with isotonic calibration for
    well-calibrated probability estimates.
    """

    def __init__(self):
        self.base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model = None
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the moneyline model with calibration.

        Args:
            X: Feature matrix of shape (n_samples, 13)
            y: Binary labels (1 = home won, 0 = away won)

        Returns:
            Dictionary with training metrics
        """
        # Cross-validation scores before calibration
        cv_scores = cross_val_score(self.base_model, X, y, cv=5, scoring="accuracy")

        # Train calibrated classifier
        self.model = CalibratedClassifierCV(
            self.base_model,
            method="isotonic",
            cv=5,
        )
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate training accuracy
        train_preds = self.model.predict(X)
        train_accuracy = (train_preds == y).mean()

        return {
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "train_accuracy": train_accuracy,
            "n_samples": len(y),
            "home_win_rate": y.mean(),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of home team winning.

        Args:
            X: Feature matrix of shape (n_samples, 13)

        Returns:
            Array of probabilities P(home wins)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # predict_proba returns [P(away), P(home)]
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return P(home wins)

    def predict_single(self, features: np.ndarray) -> float:
        """
        Predict probability for a single game.

        Args:
            features: Feature array of shape (13,)

        Returns:
            Probability of home team winning
        """
        X = features.reshape(1, -1)
        return self.predict_proba(X)[0]

    def save(self, path: str):
        """Save model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_fitted = True

    @classmethod
    def from_file(cls, path: str) -> "MoneylineModel":
        """Create a MoneylineModel instance from a saved file."""
        instance = cls()
        instance.load(path)
        return instance
