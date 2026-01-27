"""Spread model - predicts point differential using GradientBoostingRegressor."""

import numpy as np
import joblib
from pathlib import Path
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from app.core.constants import FEATURE_NAMES


class SpreadModel:
    """
    Regression model that predicts point differential (home - away).

    Uses GradientBoostingRegressor and tracks residual standard deviation
    to compute cover probabilities via normal distribution.
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
        Train the spread model.

        Args:
            X: Feature matrix of shape (n_samples, 13)
            y: Point differential (home_score - away_score)

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
            "avg_spread": y.mean(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict point differential.

        Args:
            X: Feature matrix of shape (n_samples, 13)

        Returns:
            Array of predicted point differentials (positive = home favored)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict(X)

    def predict_single(self, features: np.ndarray) -> float:
        """
        Predict point differential for a single game.

        Args:
            features: Feature array of shape (13,)

        Returns:
            Predicted point differential
        """
        X = features.reshape(1, -1)
        return self.predict(X)[0]

    def cover_probability(
        self,
        features: np.ndarray,
        spread_line: float,
        bet_on_home: bool = True,
    ) -> float:
        """
        Calculate probability of covering the spread.

        Args:
            features: Feature array of shape (13,)
            spread_line: The spread line (e.g., -4.5 means home favored by 4.5)
            bet_on_home: Whether betting on home team to cover

        Returns:
            Probability of covering the spread
        """
        predicted_diff = self.predict_single(features)

        if bet_on_home:
            # Home team covers if actual_diff > spread_line
            # P(actual_diff > spread_line) = P(Z > (spread_line - predicted_diff) / std)
            z_score = (spread_line - predicted_diff) / self.residual_std
            return 1 - stats.norm.cdf(z_score)
        else:
            # Away team covers if actual_diff < spread_line
            z_score = (spread_line - predicted_diff) / self.residual_std
            return stats.norm.cdf(z_score)

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
    def from_file(cls, path: str) -> "SpreadModel":
        """Create a SpreadModel instance from a saved file."""
        instance = cls()
        instance.load(path)
        return instance
