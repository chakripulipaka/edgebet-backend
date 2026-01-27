"""Totals model - predicts total points using GradientBoostingRegressor."""

import numpy as np
import joblib
from pathlib import Path
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from app.core.constants import FEATURE_NAMES


class TotalsModel:
    """
    Regression model that predicts total points (home + away).

    Uses GradientBoostingRegressor and tracks residual standard deviation
    to compute over/under probabilities via normal distribution.
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )
        self.residual_std = None
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the totals model.

        Args:
            X: Feature matrix of shape (n_samples, 13)
            y: Total points (home_score + away_score)

        Returns:
            Dictionary with training metrics
        """
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="neg_mean_absolute_error")

        # Fit model
        self.model.fit(X, y)

        # Calculate residual standard deviation for probability estimates
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.residual_std = np.std(residuals)

        self.is_fitted = True

        return {
            "cv_mae_mean": -cv_scores.mean(),
            "cv_mae_std": cv_scores.std(),
            "residual_std": self.residual_std,
            "n_samples": len(y),
            "avg_total": y.mean(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict total points.

        Args:
            X: Feature matrix of shape (n_samples, 13)

        Returns:
            Array of predicted total points
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict(X)

    def predict_single(self, features: np.ndarray) -> float:
        """
        Predict total points for a single game.

        Args:
            features: Feature array of shape (13,)

        Returns:
            Predicted total points
        """
        X = features.reshape(1, -1)
        return self.predict(X)[0]

    def over_probability(
        self,
        features: np.ndarray,
        total_line: float,
    ) -> float:
        """
        Calculate probability of going over the total.

        Args:
            features: Feature array of shape (13,)
            total_line: The total line (e.g., 220.5)

        Returns:
            Probability of the game going over
        """
        predicted_total = self.predict_single(features)

        # P(actual_total > line) = P(Z > (line - predicted) / std)
        z_score = (total_line - predicted_total) / self.residual_std
        return 1 - stats.norm.cdf(z_score)

    def under_probability(
        self,
        features: np.ndarray,
        total_line: float,
    ) -> float:
        """
        Calculate probability of going under the total.

        Args:
            features: Feature array of shape (13,)
            total_line: The total line (e.g., 220.5)

        Returns:
            Probability of the game going under
        """
        return 1 - self.over_probability(features, total_line)

    def save(self, path: str):
        """Save model and residual_std to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "residual_std": self.residual_std,
        }, path)

    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.residual_std = data["residual_std"]
        self.is_fitted = True

    @classmethod
    def from_file(cls, path: str) -> "TotalsModel":
        """Create a TotalsModel instance from a saved file."""
        instance = cls()
        instance.load(path)
        return instance
