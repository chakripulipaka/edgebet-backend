"""Training orchestration for all ML models."""

import logging
from pathlib import Path

from app.config import settings
from app.ml.features import FeatureBuilder
from app.ml.models.moneyline import MoneylineModel
from app.ml.models.spread import SpreadModel
from app.ml.models.totals import TotalsModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates training of all ML models."""

    def __init__(self, stats_repository):
        self.feature_builder = FeatureBuilder(stats_repository)
        self.model_dir = Path(settings.MODEL_DIR)

    async def train_all_models(self, games: list) -> dict:
        """
        Train all three models on historical game data.

        Args:
            games: List of Game objects with final scores

        Returns:
            Dictionary with training metrics for all models
        """
        logger.info(f"Building training data from {len(games)} games...")

        X, y_ml, y_spread, y_total = await self.feature_builder.build_training_data(games)

        if X is None or len(X) == 0:
            logger.error("No training data generated")
            return {"error": "No training data"}

        logger.info(f"Generated {len(X)} training samples")

        results = {}

        # Train Moneyline model
        logger.info("Training moneyline model...")
        ml_model = MoneylineModel()
        results["moneyline"] = ml_model.train(X, y_ml)
        ml_model.save(str(self.model_dir / "moneyline.joblib"))
        logger.info(f"Moneyline CV accuracy: {results['moneyline']['cv_accuracy_mean']:.3f}")

        # Train Spread model
        logger.info("Training spread model...")
        spread_model = SpreadModel()
        results["spread"] = spread_model.train(X, y_spread)
        spread_model.save(str(self.model_dir / "spread.joblib"))
        logger.info(f"Spread CV MAE: {results['spread']['cv_mae_mean']:.2f}")

        # Train Totals model
        logger.info("Training totals model...")
        totals_model = TotalsModel()
        results["totals"] = totals_model.train(X, y_total)
        totals_model.save(str(self.model_dir / "totals.joblib"))
        logger.info(f"Totals CV MAE: {results['totals']['cv_mae_mean']:.2f}")

        return results
