"""Inference pipeline for generating predictions."""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from app.config import settings
from app.ml.features import FeatureBuilder
from app.ml.models.moneyline import MoneylineModel
from app.ml.models.spread import SpreadModel
from app.ml.models.totals import TotalsModel
from app.core.calculations import calculate_implied_prob, calculate_kelly, calculate_edge

logger = logging.getLogger(__name__)


@dataclass
class BetPrediction:
    """Represents a prediction for a specific bet type."""
    bet_type: str  # "Moneyline", "Spread", "Total"
    side: str  # Team name, "Team +/-X.X", or "Over/Under X.X"
    model_prob: float
    odds: int  # American odds
    implied_prob: float
    edge: float  # Percentage
    kelly: float  # Fraction (0-0.10)

    @property
    def is_positive_ev(self) -> bool:
        return self.edge > 0


class PredictionPipeline:
    """Generates predictions for games using trained models."""

    def __init__(self, stats_repository=None):
        self.model_dir = Path(settings.MODEL_DIR)
        self.feature_builder = FeatureBuilder(stats_repository) if stats_repository else None

        # Load models (lazy loading)
        self._ml_model = None
        self._spread_model = None
        self._totals_model = None

    @property
    def moneyline_model(self) -> MoneylineModel:
        if self._ml_model is None:
            path = self.model_dir / "moneyline.joblib"
            if path.exists():
                self._ml_model = MoneylineModel.from_file(str(path))
            else:
                raise FileNotFoundError(f"Moneyline model not found at {path}")
        return self._ml_model

    @property
    def spread_model(self) -> SpreadModel:
        if self._spread_model is None:
            path = self.model_dir / "spread.joblib"
            if path.exists():
                self._spread_model = SpreadModel.from_file(str(path))
            else:
                raise FileNotFoundError(f"Spread model not found at {path}")
        return self._spread_model

    @property
    def totals_model(self) -> TotalsModel:
        if self._totals_model is None:
            path = self.model_dir / "totals.joblib"
            if path.exists():
                self._totals_model = TotalsModel.from_file(str(path))
            else:
                raise FileNotFoundError(f"Totals model not found at {path}")
        return self._totals_model

    def predict_moneyline(
        self,
        features: np.ndarray,
        home_team: str,
        away_team: str,
        home_odds: int,
        away_odds: int,
    ) -> List[BetPrediction]:
        """
        Generate moneyline predictions for both sides.

        Args:
            features: Feature array for the matchup
            home_team: Home team name
            away_team: Away team name
            home_odds: American odds for home team
            away_odds: American odds for away team

        Returns:
            List of BetPrediction objects
        """
        home_prob = self.moneyline_model.predict_single(features)
        away_prob = 1 - home_prob

        predictions = []

        # Home team prediction
        home_implied = calculate_implied_prob(home_odds)
        home_edge = calculate_edge(home_prob, home_implied)
        home_kelly = calculate_kelly(home_prob, home_odds)
        predictions.append(BetPrediction(
            bet_type="Moneyline",
            side=home_team,
            model_prob=home_prob,
            odds=home_odds,
            implied_prob=home_implied,
            edge=home_edge,
            kelly=home_kelly,
        ))

        # Away team prediction
        away_implied = calculate_implied_prob(away_odds)
        away_edge = calculate_edge(away_prob, away_implied)
        away_kelly = calculate_kelly(away_prob, away_odds)
        predictions.append(BetPrediction(
            bet_type="Moneyline",
            side=away_team,
            model_prob=away_prob,
            odds=away_odds,
            implied_prob=away_implied,
            edge=away_edge,
            kelly=away_kelly,
        ))

        return predictions

    def predict_spread(
        self,
        features: np.ndarray,
        home_team: str,
        away_team: str,
        spread_line: float,  # Negative means home favored
        home_odds: int,
        away_odds: int,
    ) -> List[BetPrediction]:
        """
        Generate spread predictions for both sides.

        Args:
            features: Feature array for the matchup
            home_team: Home team name
            away_team: Away team name
            spread_line: Spread from home team perspective (negative = home favored)
            home_odds: American odds for home spread
            away_odds: American odds for away spread

        Returns:
            List of BetPrediction objects
        """
        home_cover_prob = self.spread_model.cover_probability(
            features, spread_line, bet_on_home=True
        )
        away_cover_prob = 1 - home_cover_prob

        predictions = []

        # Home team spread
        home_spread_str = f"+{spread_line}" if spread_line > 0 else str(spread_line)
        home_implied = calculate_implied_prob(home_odds)
        home_edge = calculate_edge(home_cover_prob, home_implied)
        home_kelly = calculate_kelly(home_cover_prob, home_odds)
        predictions.append(BetPrediction(
            bet_type="Spread",
            side=f"{home_team} {home_spread_str}",
            model_prob=home_cover_prob,
            odds=home_odds,
            implied_prob=home_implied,
            edge=home_edge,
            kelly=home_kelly,
        ))

        # Away team spread (opposite of home spread)
        away_spread = -spread_line
        away_spread_str = f"+{away_spread}" if away_spread > 0 else str(away_spread)
        away_implied = calculate_implied_prob(away_odds)
        away_edge = calculate_edge(away_cover_prob, away_implied)
        away_kelly = calculate_kelly(away_cover_prob, away_odds)
        predictions.append(BetPrediction(
            bet_type="Spread",
            side=f"{away_team} {away_spread_str}",
            model_prob=away_cover_prob,
            odds=away_odds,
            implied_prob=away_implied,
            edge=away_edge,
            kelly=away_kelly,
        ))

        return predictions

    def predict_total(
        self,
        features: np.ndarray,
        total_line: float,
        over_odds: int,
        under_odds: int,
    ) -> List[BetPrediction]:
        """
        Generate over/under predictions.

        Args:
            features: Feature array for the matchup
            total_line: The total line (e.g., 220.5)
            over_odds: American odds for over
            under_odds: American odds for under

        Returns:
            List of BetPrediction objects
        """
        over_prob = self.totals_model.over_probability(features, total_line)
        under_prob = 1 - over_prob

        predictions = []

        # Over
        over_implied = calculate_implied_prob(over_odds)
        over_edge = calculate_edge(over_prob, over_implied)
        over_kelly = calculate_kelly(over_prob, over_odds)
        predictions.append(BetPrediction(
            bet_type="Total",
            side=f"Over {total_line}",
            model_prob=over_prob,
            odds=over_odds,
            implied_prob=over_implied,
            edge=over_edge,
            kelly=over_kelly,
        ))

        # Under
        under_implied = calculate_implied_prob(under_odds)
        under_edge = calculate_edge(under_prob, under_implied)
        under_kelly = calculate_kelly(under_prob, under_odds)
        predictions.append(BetPrediction(
            bet_type="Total",
            side=f"Under {total_line}",
            model_prob=under_prob,
            odds=under_odds,
            implied_prob=under_implied,
            edge=under_edge,
            kelly=under_kelly,
        ))

        return predictions

    def get_all_predictions(
        self,
        features: np.ndarray,
        home_team: str,
        away_team: str,
        odds_data: dict,
    ) -> List[BetPrediction]:
        """
        Generate all predictions for a game.

        Args:
            features: Feature array for the matchup
            home_team: Home team name
            away_team: Away team name
            odds_data: Dictionary with odds for all bet types

        Returns:
            List of all BetPrediction objects
        """
        predictions = []

        # Moneyline predictions
        if "moneyline" in odds_data:
            ml = odds_data["moneyline"]
            predictions.extend(self.predict_moneyline(
                features,
                home_team,
                away_team,
                ml.get("home_odds", -110),
                ml.get("away_odds", -110),
            ))

        # Spread predictions
        if "spread" in odds_data:
            sp = odds_data["spread"]
            predictions.extend(self.predict_spread(
                features,
                home_team,
                away_team,
                sp.get("line", 0),
                sp.get("home_odds", -110),
                sp.get("away_odds", -110),
            ))

        # Total predictions
        if "total" in odds_data:
            tot = odds_data["total"]
            predictions.extend(self.predict_total(
                features,
                tot.get("line", 220),
                tot.get("over_odds", -110),
                tot.get("under_odds", -110),
            ))

        return predictions

    def get_best_prediction(
        self,
        features: np.ndarray,
        home_team: str,
        away_team: str,
        odds_data: dict,
    ) -> Optional[BetPrediction]:
        """
        Get the best +EV prediction for a game.

        Args:
            features: Feature array for the matchup
            home_team: Home team name
            away_team: Away team name
            odds_data: Dictionary with odds for all bet types

        Returns:
            Best BetPrediction or None if no +EV bets
        """
        predictions = self.get_all_predictions(features, home_team, away_team, odds_data)

        # Filter to positive EV only
        positive_ev = [p for p in predictions if p.is_positive_ev]

        if not positive_ev:
            return None

        # Return highest edge
        return max(positive_ev, key=lambda p: p.edge)
