"""Feature engineering for ML models using rolling 10-game statistics."""

import numpy as np
from datetime import date
from typing import List, Optional

from app.core.constants import FEATURE_NAMES, ROLLING_GAMES
from app.db.models import TeamGameStats


def calculate_rolling_averages(stats: List[TeamGameStats]) -> dict:
    """
    Calculate rolling averages from a list of game stats.

    Args:
        stats: List of TeamGameStats (should be last N games, ordered by date desc)

    Returns:
        Dictionary of average stats
    """
    if not stats:
        return None

    # Initialize accumulators
    totals = {
        "points": 0.0,
        "opponent_points": 0.0,
        "fg_pct": 0.0,
        "fg3_pct": 0.0,
        "ft_pct": 0.0,
        "assists": 0.0,
        "rebounds": 0.0,
        "blocks": 0.0,
        "steals": 0.0,
        "turnovers": 0.0,
        "pace": 0.0,
        "rest_days": 0.0,
    }
    counts = {k: 0 for k in totals}

    for stat in stats:
        if stat.points is not None:
            totals["points"] += float(stat.points)
            counts["points"] += 1
        if stat.opponent_points is not None:
            totals["opponent_points"] += float(stat.opponent_points)
            counts["opponent_points"] += 1
        if stat.fg_pct is not None:
            totals["fg_pct"] += float(stat.fg_pct)
            counts["fg_pct"] += 1
        if stat.fg3_pct is not None:
            totals["fg3_pct"] += float(stat.fg3_pct)
            counts["fg3_pct"] += 1
        if stat.ft_pct is not None:
            totals["ft_pct"] += float(stat.ft_pct)
            counts["ft_pct"] += 1
        if stat.assists is not None:
            totals["assists"] += float(stat.assists)
            counts["assists"] += 1
        if stat.rebounds is not None:
            totals["rebounds"] += float(stat.rebounds)
            counts["rebounds"] += 1
        if stat.blocks is not None:
            totals["blocks"] += float(stat.blocks)
            counts["blocks"] += 1
        if stat.steals is not None:
            totals["steals"] += float(stat.steals)
            counts["steals"] += 1
        if stat.turnovers is not None:
            totals["turnovers"] += float(stat.turnovers)
            counts["turnovers"] += 1
        if stat.pace is not None:
            totals["pace"] += float(stat.pace)
            counts["pace"] += 1
        if stat.rest_days is not None:
            totals["rest_days"] += float(stat.rest_days)
            counts["rest_days"] += 1

    # Calculate averages
    averages = {}
    for key in totals:
        if counts[key] > 0:
            averages[key] = totals[key] / counts[key]
        else:
            averages[key] = 0.0

    return averages


def compute_differential_features(
    home_rolling: dict,
    away_rolling: dict,
) -> np.ndarray:
    """
    Compute differential features for a matchup.

    Args:
        home_rolling: Rolling averages for home team
        away_rolling: Rolling averages for away team

    Returns:
        Feature array of shape (13,) matching FEATURE_NAMES order
    """
    if not home_rolling or not away_rolling:
        return None

    features = np.array([
        home_rolling["points"] - away_rolling["points"],             # points_diff
        home_rolling["opponent_points"] - away_rolling["opponent_points"],  # opponent_points_diff (defensive)
        home_rolling["fg_pct"] - away_rolling["fg_pct"],             # fg_pct_diff
        home_rolling["fg3_pct"] - away_rolling["fg3_pct"],           # fg3_pct_diff
        home_rolling["ft_pct"] - away_rolling["ft_pct"],             # ft_pct_diff
        home_rolling["assists"] - away_rolling["assists"],           # assists_diff
        home_rolling["rebounds"] - away_rolling["rebounds"],         # rebounds_diff
        home_rolling["blocks"] - away_rolling["blocks"],             # blocks_diff
        home_rolling["steals"] - away_rolling["steals"],             # steals_diff
        home_rolling["turnovers"] - away_rolling["turnovers"],       # turnovers_diff
        home_rolling["pace"] - away_rolling["pace"],                 # pace_diff
        home_rolling["rest_days"] - away_rolling["rest_days"],       # rest_days_diff
        1.0,  # is_home (always 1, from home team perspective)
    ])

    return features


def compute_features_from_raw_stats(
    home_stats: list[dict],
    away_stats: list[dict],
) -> Optional[np.ndarray]:
    """
    Compute features from raw stat dictionaries (for API-fetched data).

    Args:
        home_stats: List of home team's recent game stats
        away_stats: List of away team's recent game stats

    Returns:
        Feature array or None if insufficient data
    """
    if len(home_stats) < 5 or len(away_stats) < 5:
        return None

    def calc_averages(stats: list[dict]) -> dict:
        if not stats:
            return None
        n = len(stats)
        return {
            "points": sum(s.get("points", 0) or 0 for s in stats) / n,
            "opponent_points": sum(s.get("opponent_points", 0) or 0 for s in stats) / n,
            "fg_pct": sum(s.get("fg_pct", 0) or 0 for s in stats) / n,
            "fg3_pct": sum(s.get("fg3_pct", 0) or 0 for s in stats) / n,
            "ft_pct": sum(s.get("ft_pct", 0) or 0 for s in stats) / n,
            "assists": sum(s.get("assists", 0) or 0 for s in stats) / n,
            "rebounds": sum(s.get("rebounds", 0) or 0 for s in stats) / n,
            "blocks": sum(s.get("blocks", 0) or 0 for s in stats) / n,
            "steals": sum(s.get("steals", 0) or 0 for s in stats) / n,
            "turnovers": sum(s.get("turnovers", 0) or 0 for s in stats) / n,
            "pace": sum(s.get("pace", 100) or 100 for s in stats) / n,
            "rest_days": sum(s.get("rest_days", 2) or 2 for s in stats) / n,
        }

    home_avg = calc_averages(home_stats)
    away_avg = calc_averages(away_stats)

    if not home_avg or not away_avg:
        return None

    return compute_differential_features(home_avg, away_avg)


class FeatureBuilder:
    """Builds feature matrices for training and inference."""

    def __init__(self, stats_repository):
        self.stats_repo = stats_repository

    async def build_features_for_game(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: date,
    ) -> Optional[np.ndarray]:
        """
        Build features for a specific game.

        Args:
            home_team_id: Database ID of home team
            away_team_id: Database ID of away team
            game_date: Date of the game

        Returns:
            Feature array or None if insufficient data
        """
        # Get rolling stats for both teams
        home_stats = await self.stats_repo.get_team_rolling_stats(
            home_team_id, game_date, ROLLING_GAMES
        )
        away_stats = await self.stats_repo.get_team_rolling_stats(
            away_team_id, game_date, ROLLING_GAMES
        )

        if len(home_stats) < 5 or len(away_stats) < 5:
            return None

        home_rolling = calculate_rolling_averages(home_stats)
        away_rolling = calculate_rolling_averages(away_stats)

        return compute_differential_features(home_rolling, away_rolling)

    async def build_training_data(
        self,
        games: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build training datasets from historical games.

        Args:
            games: List of Game objects with scores

        Returns:
            Tuple of (X, y_moneyline, y_spread, y_total)
        """
        X_list = []
        y_ml_list = []    # 1 if home won, 0 if away won
        y_spread_list = []  # Point differential (positive = home won by that much)
        y_total_list = []   # Total points

        for game in games:
            if game.home_score is None or game.away_score is None:
                continue

            features = await self.build_features_for_game(
                game.home_team_id,
                game.away_team_id,
                game.game_date,
            )

            if features is None:
                continue

            X_list.append(features)
            y_ml_list.append(1 if game.home_score > game.away_score else 0)
            y_spread_list.append(game.home_score - game.away_score)
            y_total_list.append(game.home_score + game.away_score)

        if not X_list:
            return None, None, None, None

        return (
            np.array(X_list),
            np.array(y_ml_list),
            np.array(y_spread_list),
            np.array(y_total_list),
        )
