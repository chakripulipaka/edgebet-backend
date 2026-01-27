"""CLI command to predict today's NBA games using trained ML models."""

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import stats

from app.config import settings
from app.core.calculations import calculate_implied_prob
from app.core.constants import NBA_TEAM_MAP, ROLLING_GAMES
from app.services.espn_data import espn_data_service
from app.services.odds_data import odds_service
from app.ml.models.moneyline import MoneylineModel
from app.ml.models.spread import SpreadModel
from app.ml.models.totals import TotalsModel

logger = logging.getLogger(__name__)


def get_team_abbreviation(team_id: int) -> str:
    """Get team abbreviation from NBA team ID."""
    team_info = NBA_TEAM_MAP.get(team_id, {})
    return team_info.get("abbreviation", f"TEAM_{team_id}")


def get_team_recent_stats(team_id: int, season: str, num_games: int = 10) -> List[Dict]:
    """
    Fetch last N games with full box score stats for a team.

    Since get_team_game_logs() returns most stats as None, we need to:
    1. Get team's schedule to find completed game IDs
    2. Call get_box_score(game_id) for each game to get full stats

    Args:
        team_id: NBA team ID
        season: Season string (e.g., "2024-25")
        num_games: Number of recent games to fetch

    Returns:
        List of stat dictionaries for each game
    """
    # Get recent game logs (will have game_id and game_date, but stats are None)
    game_logs = espn_data_service.get_team_game_logs(team_id, season, num_games)

    if not game_logs:
        logger.warning(f"No game logs found for team {team_id}")
        return []

    stats_list = []

    for game_log in game_logs:
        game_id = game_log.get("game_id")
        game_date = game_log.get("game_date")

        if not game_id:
            continue

        # Fetch full box score for this game
        box_score = espn_data_service.get_box_score(game_id)

        if not box_score:
            logger.debug(f"No box score available for game {game_id}")
            continue

        # Get this team's stats from the box score
        team_stats = box_score.get(team_id, {})

        if not team_stats or team_stats.get("points") is None:
            logger.debug(f"No stats for team {team_id} in game {game_id}")
            continue

        # Get opponent stats for opponent_points
        opponent_id = None
        for other_team_id in box_score.keys():
            if other_team_id != team_id:
                opponent_id = other_team_id
                break

        opponent_stats = box_score.get(opponent_id, {}) if opponent_id else {}

        stats = {
            "game_id": game_id,
            "game_date": game_date,
            "points": team_stats.get("points"),
            "opponent_points": opponent_stats.get("points"),
            "fg_pct": team_stats.get("fg_pct"),
            "fg3_pct": team_stats.get("fg3_pct"),
            "ft_pct": team_stats.get("ft_pct"),
            "assists": team_stats.get("assists"),
            "rebounds": team_stats.get("rebounds"),
            "blocks": team_stats.get("blocks"),
            "steals": team_stats.get("steals"),
            "turnovers": team_stats.get("turnovers"),
        }

        stats_list.append(stats)

    logger.debug(f"Fetched {len(stats_list)} games with stats for team {team_id}")
    return stats_list


def compute_rolling_averages(stats_list: List[Dict]) -> Dict[str, float]:
    """
    Compute rolling averages from a list of game stats.

    Args:
        stats_list: List of stat dictionaries from recent games

    Returns:
        Dictionary with rolling averages for each stat
    """
    if not stats_list:
        return {}

    stat_keys = [
        "points", "opponent_points", "fg_pct", "fg3_pct", "ft_pct",
        "assists", "rebounds", "blocks", "steals", "turnovers"
    ]

    averages = {}

    for key in stat_keys:
        values = [s.get(key) for s in stats_list if s.get(key) is not None]
        if values:
            averages[key] = sum(values) / len(values)
        else:
            averages[key] = 0.0

    # Calculate average rest days from game dates
    if len(stats_list) >= 2:
        dates = [s.get("game_date") for s in stats_list if s.get("game_date")]
        if len(dates) >= 2:
            # Sort dates descending (most recent first)
            dates = sorted(dates, reverse=True)
            rest_days_list = []
            for i in range(len(dates) - 1):
                diff = (dates[i] - dates[i + 1]).days - 1
                rest_days_list.append(max(0, min(diff, 7)))  # Cap at 7
            averages["rest_days"] = sum(rest_days_list) / len(rest_days_list)
        else:
            averages["rest_days"] = 2.0
    else:
        averages["rest_days"] = 2.0  # Default

    return averages


def build_features(home_rolling: Dict[str, float], away_rolling: Dict[str, float]) -> np.ndarray:
    """
    Build 13-feature array for the moneyline model.

    Features match the training data format:
    1. points_diff
    2. opponent_points_diff
    3. fg_pct_diff
    4. fg3_pct_diff
    5. ft_pct_diff
    6. assists_diff
    7. rebounds_diff
    8. blocks_diff
    9. steals_diff
    10. turnovers_diff
    11. pace_diff (not available, use 0)
    12. rest_days_diff
    13. is_home (always 1)

    Args:
        home_rolling: Rolling averages for home team
        away_rolling: Rolling averages for away team

    Returns:
        NumPy array of shape (13,)
    """
    features = np.array([
        home_rolling.get("points", 0) - away_rolling.get("points", 0),
        home_rolling.get("opponent_points", 0) - away_rolling.get("opponent_points", 0),
        home_rolling.get("fg_pct", 0) - away_rolling.get("fg_pct", 0),
        home_rolling.get("fg3_pct", 0) - away_rolling.get("fg3_pct", 0),
        home_rolling.get("ft_pct", 0) - away_rolling.get("ft_pct", 0),
        home_rolling.get("assists", 0) - away_rolling.get("assists", 0),
        home_rolling.get("rebounds", 0) - away_rolling.get("rebounds", 0),
        home_rolling.get("blocks", 0) - away_rolling.get("blocks", 0),
        home_rolling.get("steals", 0) - away_rolling.get("steals", 0),
        home_rolling.get("turnovers", 0) - away_rolling.get("turnovers", 0),
        0.0,  # pace_diff (not available from ESPN)
        home_rolling.get("rest_days", 2) - away_rolling.get("rest_days", 2),
        1.0,  # is_home
    ])

    return features


def load_models() -> Tuple[Optional[MoneylineModel], Optional[SpreadModel], Optional[TotalsModel]]:
    """Load all trained models from disk."""
    model_dir = Path(settings.MODEL_DIR)

    moneyline_model = None
    spread_model = None
    totals_model = None

    # Load moneyline model
    moneyline_path = model_dir / "moneyline.joblib"
    if moneyline_path.exists():
        try:
            moneyline_model = MoneylineModel.from_file(str(moneyline_path))
            logger.info(f"Loaded moneyline model from {moneyline_path}")
        except Exception as e:
            logger.error(f"Failed to load moneyline model: {e}")
    else:
        logger.error(f"Moneyline model not found at {moneyline_path}")

    # Load spread model
    spread_path = model_dir / "spread.joblib"
    if spread_path.exists():
        try:
            spread_model = SpreadModel.from_file(str(spread_path))
            logger.info(f"Loaded spread model (residual_std={spread_model.residual_std:.1f})")
        except Exception as e:
            logger.error(f"Failed to load spread model: {e}")
    else:
        logger.warning(f"Spread model not found at {spread_path}")

    # Load totals model
    totals_path = model_dir / "totals.joblib"
    if totals_path.exists():
        try:
            totals_model = TotalsModel.from_file(str(totals_path))
            logger.info(f"Loaded totals model (residual_std={totals_model.residual_std:.1f})")
        except Exception as e:
            logger.error(f"Failed to load totals model: {e}")
    else:
        logger.warning(f"Totals model not found at {totals_path}")

    if moneyline_model is None:
        logger.error("Run 'python -m cli.commands train-models' first")

    return moneyline_model, spread_model, totals_model


def home_win_prob_from_spread(predicted_spread: float, residual_std: float) -> float:
    """
    Calculate P(home wins) from spread prediction using normal distribution.

    Args:
        predicted_spread: Predicted point differential (positive = home favored)
        residual_std: Model uncertainty (standard deviation of residuals)

    Returns:
        Probability that home team wins (actual_spread > 0)
    """
    z_score = predicted_spread / residual_std
    return stats.norm.cdf(z_score)


def cover_prob_from_spread(predicted_spread: float, line: float, residual_std: float) -> Tuple[float, float]:
    """
    Calculate cover probabilities for both sides given a spread line.

    Args:
        predicted_spread: Predicted point differential (positive = home favored)
        line: Vegas spread line (negative = home favored, e.g., -4.5)
        residual_std: Model uncertainty

    Returns:
        Tuple of (P(home covers), P(away covers))
    """
    # Home covers if actual_spread > line
    z_score = (line - predicted_spread) / residual_std
    home_cover = 1 - stats.norm.cdf(z_score)
    away_cover = stats.norm.cdf(z_score)
    return home_cover, away_cover


def over_under_prob(predicted_total: float, line: float, residual_std: float) -> Tuple[float, float]:
    """
    Calculate over/under probabilities given a total line.

    Args:
        predicted_total: Predicted total points
        line: Vegas O/U line (e.g., 221.5)
        residual_std: Model uncertainty

    Returns:
        Tuple of (P(over), P(under))
    """
    z_score = (line - predicted_total) / residual_std
    over_prob = 1 - stats.norm.cdf(z_score)
    under_prob = stats.norm.cdf(z_score)
    return over_prob, under_prob


def format_spread(spread: float, home_abbr: str, away_abbr: str) -> str:
    """Format spread for display from home team perspective (e.g., 'BOS -5.2' or 'POR +3.5')."""
    if spread == 0:
        return "PICK"
    sign = "+" if spread > 0 else ""
    return f"{home_abbr} {sign}{spread:.1f}"


def predict_today(target_date: Optional[date] = None):
    """
    Predict today's NBA games using all trained ML models.

    Workflow:
    1. Get today's games from ESPN
    2. For each game, get last 10 games per team via box scores
    3. Compute rolling averages
    4. Build differential features
    5. Load all models and predict
    6. Print results with statistical probabilities

    Args:
        target_date: Date to predict (default: today)
    """
    if target_date is None:
        target_date = date.today()

    # Determine season (NBA season spans two calendar years)
    # e.g., games in Jan 2025 are part of 2024-25 season
    if target_date.month >= 10:
        season = f"{target_date.year}-{str(target_date.year + 1)[-2:]}"
    else:
        season = f"{target_date.year - 1}-{str(target_date.year)[-2:]}"

    print(f"\nFetching games for {target_date}...")

    # 1. Get today's games
    games = espn_data_service.get_games_for_date(target_date)

    if not games:
        print(f"No games found for {target_date}")
        return

    # Filter to scheduled games only (not yet played)
    scheduled_games = [g for g in games if g.get("status") in ("scheduled", None)]

    if not scheduled_games:
        # If no scheduled games, show all games for the day
        scheduled_games = games

    print(f"Found {len(scheduled_games)} games for {target_date}\n")

    # 2. Fetch live odds from The Odds API
    print("Fetching Vegas odds...", end=" ", flush=True)
    try:
        odds_data = asyncio.run(odds_service.get_nba_odds(target_date))
        if odds_data:
            print(f"done ({len(odds_data)} games)")
        else:
            print("unavailable (no API key or no games found)")
    except Exception as e:
        logger.warning(f"Failed to fetch odds: {e}")
        odds_data = []
        print("failed")
    print()

    # 3. Load all models
    moneyline_model, spread_model, totals_model = load_models()
    if moneyline_model is None:
        return

    # 4. Process each game
    for game in scheduled_games:
        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        home_abbr = get_team_abbreviation(home_team_id)
        away_abbr = get_team_abbreviation(away_team_id)

        print(f"{'='*50}")
        print(f"{away_abbr} @ {home_abbr}")
        print(f"{'='*50}")

        # Get last 10 games for each team
        print(f"  Fetching stats...", end=" ", flush=True)
        home_stats = get_team_recent_stats(home_team_id, season, ROLLING_GAMES)
        away_stats = get_team_recent_stats(away_team_id, season, ROLLING_GAMES)
        print(f"done ({len(home_stats)}/{len(away_stats)} games)")

        # Check if we have enough data
        min_games = 5
        if len(home_stats) < min_games or len(away_stats) < min_games:
            print(f"  Insufficient data (need at least {min_games} games per team)")
            print()
            continue

        # Compute rolling averages
        home_rolling = compute_rolling_averages(home_stats)
        away_rolling = compute_rolling_averages(away_stats)

        # Build features
        features = build_features(home_rolling, away_rolling)

        # Match game to odds data
        game_odds = odds_service.match_game_to_odds(home_abbr, away_abbr, odds_data)

        # ========== MONEYLINE ==========
        print(f"\n  MONEYLINE")
        home_win_prob = moneyline_model.predict_single(features)
        away_win_prob = 1 - home_win_prob
        print(f"    Model: {home_win_prob:.1%} {home_abbr} / {away_win_prob:.1%} {away_abbr}")

        # Get moneyline odds from matched game
        ml_odds = game_odds.get("moneyline") if game_odds else None

        if ml_odds:
            home_odds = ml_odds.get("home_odds", -110)
            away_odds = ml_odds.get("away_odds", -110)

            # Format odds with +/- sign
            home_odds_str = f"{home_odds:+d}" if home_odds > 0 else str(home_odds)
            away_odds_str = f"{away_odds:+d}" if away_odds > 0 else str(away_odds)

            # Calculate implied probabilities from Vegas odds
            vegas_home_prob = calculate_implied_prob(home_odds)
            vegas_away_prob = calculate_implied_prob(away_odds)

            print(f"    Vegas: {home_abbr} {home_odds_str} ({vegas_home_prob:.1%}) / {away_abbr} {away_odds_str} ({vegas_away_prob:.1%})")

            # Calculate edge for each side (model prob - vegas implied prob)
            home_edge = (home_win_prob - vegas_home_prob) * 100
            away_edge = (away_win_prob - vegas_away_prob) * 100

            # Recommend the side with positive edge (where model beats Vegas)
            if home_edge > away_edge:
                # Home has better edge
                if home_edge > 0:
                    print(f"    >>> BET {home_abbr}: +{home_edge:.1f}% edge (model: {home_win_prob:.1%}, Vegas: {vegas_home_prob:.1%})")
                else:
                    print(f"    >>> No edge (best: {home_abbr} at {home_edge:.1f}%)")
            else:
                # Away has better edge
                if away_edge > 0:
                    print(f"    >>> BET {away_abbr}: +{away_edge:.1f}% edge (model: {away_win_prob:.1%}, Vegas: {vegas_away_prob:.1%})")
                else:
                    print(f"    >>> No edge (best: {away_abbr} at {away_edge:.1f}%)")
        else:
            # No Vegas odds available - just show model prediction
            print(f"    Vegas: unavailable")
            if home_win_prob > 0.5:
                print(f"    >>> Model favors {home_abbr} ({home_win_prob:.1%})")
            else:
                print(f"    >>> Model favors {away_abbr} ({away_win_prob:.1%})")

        # ========== SPREAD ==========
        if spread_model is not None:
            print(f"\n  SPREAD")
            predicted_spread = spread_model.predict_single(features)
            residual_std = spread_model.residual_std

            # Format spread for display
            spread_str = format_spread(predicted_spread, home_abbr, away_abbr)
            print(f"    Model prediction: {spread_str}")

            # Get spread odds from matched game
            spread_odds = game_odds.get("spread") if game_odds else None

            if spread_odds:
                vegas_line = spread_odds.get("line", 0)
                # Format Vegas line for display
                vegas_str = format_spread(vegas_line, home_abbr, away_abbr)
                print(f"    Vegas line: {vegas_str}")

                # Calculate edge (difference between model and Vegas)
                edge = predicted_spread - vegas_line
                if edge > 0:
                    edge_str = f"{edge:.1f} pts (model more bullish on {away_abbr})"
                elif edge < 0:
                    edge_str = f"{abs(edge):.1f} pts (model more bullish on {home_abbr})"
                else:
                    edge_str = "0.0 pts (model agrees with Vegas)"
                print(f"    Edge: {edge_str}")

                # Show P(line hits) for the direction model predicts
                home_cover, away_cover = cover_prob_from_spread(predicted_spread, vegas_line, residual_std)
                if predicted_spread > vegas_line:
                    # Model predicts home covers
                    print(f"    >>> P({home_abbr} {vegas_line:+.1f} hits): {home_cover:.1%}")
                else:
                    # Model predicts away covers
                    print(f"    >>> P({away_abbr} {-vegas_line:+.1f} hits): {away_cover:.1%}")
            else:
                print(f"    Vegas line: unavailable")

        # ========== TOTAL ==========
        if totals_model is not None:
            print(f"\n  TOTAL")
            predicted_total = totals_model.predict_single(features)
            total_std = totals_model.residual_std

            print(f"    Model prediction: {predicted_total:.1f} pts")

            # Get total odds from matched game
            total_odds = game_odds.get("total") if game_odds else None

            if total_odds:
                vegas_total = total_odds.get("line", 0)
                print(f"    Vegas line: O/U {vegas_total}")

                # Calculate edge
                edge = predicted_total - vegas_total
                if edge > 0:
                    edge_str = f"{edge:.1f} pts over"
                elif edge < 0:
                    edge_str = f"{abs(edge):.1f} pts under"
                else:
                    edge_str = "0.0 pts (model agrees with Vegas)"
                print(f"    Edge: {edge_str}")

                # Show P(Over/Under hits) for the direction model predicts
                over_prob, under_prob = over_under_prob(predicted_total, vegas_total, total_std)
                if predicted_total > vegas_total:
                    # Model predicts over
                    print(f"    >>> P(Over {vegas_total} hits): {over_prob:.1%}")
                else:
                    # Model predicts under
                    print(f"    >>> P(Under {vegas_total} hits): {under_prob:.1%}")
            else:
                print(f"    Vegas line: unavailable")

        print()

    print("Predictions complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    predict_today()
