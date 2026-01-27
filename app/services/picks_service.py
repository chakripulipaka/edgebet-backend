"""Service for generating and managing daily picks."""

import logging
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.repositories.picks import PicksRepository
from app.db.models import DailyPick
from app.services.espn_data import espn_data_service
from app.services.odds_data import odds_service
from app.core.constants import NBA_TEAM_MAP, ROLLING_GAMES
from app.core.calculations import calculate_implied_prob, calculate_kelly, calculate_edge
from app.ml.models.moneyline import MoneylineModel
from app.ml.models.spread import SpreadModel
from app.ml.models.totals import TotalsModel

logger = logging.getLogger(__name__)


def get_team_name(team_id: int) -> str:
    """Get full team name from NBA team ID."""
    team_info = NBA_TEAM_MAP.get(team_id, {})
    city = team_info.get("city", "")
    name = team_info.get("name", f"Team {team_id}")
    return f"{city} {name}".strip()


def get_team_abbreviation(team_id: int) -> str:
    """Get team abbreviation from NBA team ID."""
    team_info = NBA_TEAM_MAP.get(team_id, {})
    return team_info.get("abbreviation", f"TEAM_{team_id}")


def get_team_recent_stats(team_id: int, season: str, num_games: int = 10) -> List[Dict]:
    """
    Fetch last N games with full box score stats for a team.

    Args:
        team_id: NBA team ID
        season: Season string (e.g., "2024-25")
        num_games: Number of recent games to fetch

    Returns:
        List of stat dictionaries for each game
    """
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

        box_score = espn_data_service.get_box_score(game_id)

        if not box_score:
            continue

        team_stats = box_score.get(team_id, {})

        if not team_stats or team_stats.get("points") is None:
            continue

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
    """Compute rolling averages from a list of game stats."""
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

    if len(stats_list) >= 2:
        dates = [s.get("game_date") for s in stats_list if s.get("game_date")]
        if len(dates) >= 2:
            dates = sorted(dates, reverse=True)
            rest_days_list = []
            for i in range(len(dates) - 1):
                diff = (dates[i] - dates[i + 1]).days - 1
                rest_days_list.append(max(0, min(diff, 7)))
            averages["rest_days"] = sum(rest_days_list) / len(rest_days_list)
        else:
            averages["rest_days"] = 2.0
    else:
        averages["rest_days"] = 2.0

    return averages


def build_features(home_rolling: Dict[str, float], away_rolling: Dict[str, float]) -> np.ndarray:
    """Build 13-feature array for the ML models."""
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

    moneyline_path = model_dir / "moneyline.joblib"
    if moneyline_path.exists():
        try:
            moneyline_model = MoneylineModel.from_file(str(moneyline_path))
            logger.info(f"Loaded moneyline model from {moneyline_path}")
        except Exception as e:
            logger.error(f"Failed to load moneyline model: {e}")

    spread_path = model_dir / "spread.joblib"
    if spread_path.exists():
        try:
            spread_model = SpreadModel.from_file(str(spread_path))
            logger.info(f"Loaded spread model")
        except Exception as e:
            logger.error(f"Failed to load spread model: {e}")

    totals_path = model_dir / "totals.joblib"
    if totals_path.exists():
        try:
            totals_model = TotalsModel.from_file(str(totals_path))
            logger.info(f"Loaded totals model")
        except Exception as e:
            logger.error(f"Failed to load totals model: {e}")

    return moneyline_model, spread_model, totals_model


def cover_prob_from_spread(predicted_spread: float, line: float, residual_std: float) -> Tuple[float, float]:
    """Calculate cover probabilities for both sides given a spread line."""
    z_score = (line - predicted_spread) / residual_std
    home_cover = 1 - stats.norm.cdf(z_score)
    away_cover = stats.norm.cdf(z_score)
    return home_cover, away_cover


def over_under_prob(predicted_total: float, line: float, residual_std: float) -> Tuple[float, float]:
    """Calculate over/under probabilities given a total line."""
    z_score = (line - predicted_total) / residual_std
    over_prob = 1 - stats.norm.cdf(z_score)
    under_prob = stats.norm.cdf(z_score)
    return over_prob, under_prob


class PicksService:
    """Service for generating and retrieving betting picks."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.picks_repo = PicksRepository(session)

    async def get_todays_picks(self) -> List[dict]:
        """Get today's picks formatted for the API response."""
        today = date.today()
        return await self.generate_picks_for_date(today)

    async def generate_picks_for_date(self, pick_date: date) -> List[dict]:
        """
        Generate picks for all games on a given date.

        1. Check if picks already exist for this date in the database
        2. If yes, return cached picks (filtered by game status)
        3. If no, fetch games from ESPN, generate predictions, save to DB
        """
        # Check for existing picks first
        existing_picks = await self.picks_repo.get_all_by_date(pick_date)
        if existing_picks:
            logger.info(f"Found {len(existing_picks)} cached picks for {pick_date}")
            # Get current game statuses from ESPN
            game_statuses = self._get_game_statuses(pick_date)
            return self._format_picks_with_status(existing_picks, game_statuses)

        # Determine season
        if pick_date.month >= 10:
            season = f"{pick_date.year}-{str(pick_date.year + 1)[-2:]}"
        else:
            season = f"{pick_date.year - 1}-{str(pick_date.year)[-2:]}"

        # Fetch games from ESPN
        logger.info(f"Fetching games from ESPN for {pick_date}")
        games = espn_data_service.get_games_for_date(pick_date)

        if not games:
            logger.info(f"No games found for {pick_date}")
            return []

        # Filter to scheduled games
        scheduled_games = [g for g in games if g.get("status") in ("scheduled", None)]
        if not scheduled_games:
            scheduled_games = games

        logger.info(f"Found {len(scheduled_games)} games for {pick_date}")

        # Fetch odds
        try:
            odds_data = await odds_service.get_nba_odds(pick_date)
            logger.info(f"Fetched odds for {len(odds_data) if odds_data else 0} games")
        except Exception as e:
            logger.warning(f"Failed to fetch odds: {e}")
            odds_data = []

        # Load models
        moneyline_model, spread_model, totals_model = load_models()
        if moneyline_model is None:
            logger.error("No moneyline model available - cannot generate predictions")
            return []

        # Generate picks for each game
        all_picks = []
        for game in scheduled_games:
            game_picks = await self._generate_picks_for_game(
                game, season, odds_data,
                moneyline_model, spread_model, totals_model,
                pick_date
            )
            all_picks.extend(game_picks)

        # Sort by edge descending
        all_picks.sort(key=lambda p: p.get("edge", 0), reverse=True)

        return all_picks

    async def _generate_picks_for_game(
        self,
        game: Dict,
        season: str,
        odds_data: List[dict],
        moneyline_model: MoneylineModel,
        spread_model: Optional[SpreadModel],
        totals_model: Optional[TotalsModel],
        pick_date: date,
    ) -> List[dict]:
        """Generate predictions for a single game and save positive EV picks."""
        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]
        espn_game_id = game["nba_game_id"]

        home_name = get_team_name(home_team_id)
        away_name = get_team_name(away_team_id)
        home_abbr = get_team_abbreviation(home_team_id)
        away_abbr = get_team_abbreviation(away_team_id)

        logger.info(f"Processing game: {away_abbr} @ {home_abbr}")

        # Get recent stats for each team
        home_stats = get_team_recent_stats(home_team_id, season, ROLLING_GAMES)
        away_stats = get_team_recent_stats(away_team_id, season, ROLLING_GAMES)

        min_games = 5
        if len(home_stats) < min_games or len(away_stats) < min_games:
            logger.warning(f"Insufficient data for {away_abbr} @ {home_abbr}")
            return []

        # Compute rolling averages and build features
        home_rolling = compute_rolling_averages(home_stats)
        away_rolling = compute_rolling_averages(away_stats)
        features = build_features(home_rolling, away_rolling)

        # Match game to odds
        game_odds = odds_service.match_game_to_odds(home_abbr, away_abbr, odds_data)
        if not game_odds:
            game_odds = self._default_odds()

        picks = []

        # MONEYLINE predictions
        home_win_prob = moneyline_model.predict_single(features)
        away_win_prob = 1 - home_win_prob

        ml_odds = game_odds.get("moneyline", {})
        home_ml_odds = ml_odds.get("home_odds", -110)
        away_ml_odds = ml_odds.get("away_odds", -110)

        # Check home ML
        home_implied = calculate_implied_prob(home_ml_odds)
        home_edge = calculate_edge(home_win_prob, home_implied)
        if home_edge > 0:
            kelly = calculate_kelly(home_win_prob, home_ml_odds)
            pick = await self._save_and_format_pick(
                pick_date=pick_date,
                espn_game_id=espn_game_id,
                home_team=home_name,
                away_team=away_name,
                game_time=game.get("game_time"),
                bet_type="Moneyline",
                side=home_name,
                odds=home_ml_odds,
                model_prob=home_win_prob,
                implied_prob=home_implied,
                edge=home_edge,
                kelly_capped=kelly,
            )
            picks.append(pick)

        # Check away ML
        away_implied = calculate_implied_prob(away_ml_odds)
        away_edge = calculate_edge(away_win_prob, away_implied)
        if away_edge > 0:
            kelly = calculate_kelly(away_win_prob, away_ml_odds)
            pick = await self._save_and_format_pick(
                pick_date=pick_date,
                espn_game_id=espn_game_id,
                home_team=home_name,
                away_team=away_name,
                game_time=game.get("game_time"),
                bet_type="Moneyline",
                side=away_name,
                odds=away_ml_odds,
                model_prob=away_win_prob,
                implied_prob=away_implied,
                edge=away_edge,
                kelly_capped=kelly,
            )
            picks.append(pick)

        # SPREAD predictions
        if spread_model is not None:
            predicted_spread = spread_model.predict_single(features)
            residual_std = spread_model.residual_std

            spread_odds = game_odds.get("spread", {})
            vegas_line = spread_odds.get("line", 0)
            home_spread_odds = spread_odds.get("home_odds", -110)
            away_spread_odds = spread_odds.get("away_odds", -110)

            home_cover, away_cover = cover_prob_from_spread(predicted_spread, vegas_line, residual_std)

            # Check home spread
            home_spread_implied = calculate_implied_prob(home_spread_odds)
            home_spread_edge = calculate_edge(home_cover, home_spread_implied)
            if home_spread_edge > 0:
                kelly = calculate_kelly(home_cover, home_spread_odds)
                side = f"{home_name} {vegas_line:+.1f}"
                pick = await self._save_and_format_pick(
                    pick_date=pick_date,
                    espn_game_id=espn_game_id,
                    home_team=home_name,
                    away_team=away_name,
                    game_time=game.get("game_time"),
                    bet_type="Spread",
                    side=side,
                    odds=home_spread_odds,
                    model_prob=home_cover,
                    implied_prob=home_spread_implied,
                    edge=home_spread_edge,
                    kelly_capped=kelly,
                )
                picks.append(pick)

            # Check away spread
            away_spread_implied = calculate_implied_prob(away_spread_odds)
            away_spread_edge = calculate_edge(away_cover, away_spread_implied)
            if away_spread_edge > 0:
                kelly = calculate_kelly(away_cover, away_spread_odds)
                side = f"{away_name} {-vegas_line:+.1f}"
                pick = await self._save_and_format_pick(
                    pick_date=pick_date,
                    espn_game_id=espn_game_id,
                    home_team=home_name,
                    away_team=away_name,
                    game_time=game.get("game_time"),
                    bet_type="Spread",
                    side=side,
                    odds=away_spread_odds,
                    model_prob=away_cover,
                    implied_prob=away_spread_implied,
                    edge=away_spread_edge,
                    kelly_capped=kelly,
                )
                picks.append(pick)

        # TOTAL predictions
        if totals_model is not None:
            predicted_total = totals_model.predict_single(features)
            total_std = totals_model.residual_std

            total_odds = game_odds.get("total", {})
            vegas_total = total_odds.get("line", 220)
            over_odds = total_odds.get("over_odds", -110)
            under_odds = total_odds.get("under_odds", -110)

            over_prob, under_prob = over_under_prob(predicted_total, vegas_total, total_std)

            # Check over
            over_implied = calculate_implied_prob(over_odds)
            over_edge = calculate_edge(over_prob, over_implied)
            if over_edge > 0:
                kelly = calculate_kelly(over_prob, over_odds)
                side = f"Over {vegas_total}"
                pick = await self._save_and_format_pick(
                    pick_date=pick_date,
                    espn_game_id=espn_game_id,
                    home_team=home_name,
                    away_team=away_name,
                    game_time=game.get("game_time"),
                    bet_type="Total",
                    side=side,
                    odds=over_odds,
                    model_prob=over_prob,
                    implied_prob=over_implied,
                    edge=over_edge,
                    kelly_capped=kelly,
                )
                picks.append(pick)

            # Check under
            under_implied = calculate_implied_prob(under_odds)
            under_edge = calculate_edge(under_prob, under_implied)
            if under_edge > 0:
                kelly = calculate_kelly(under_prob, under_odds)
                side = f"Under {vegas_total}"
                pick = await self._save_and_format_pick(
                    pick_date=pick_date,
                    espn_game_id=espn_game_id,
                    home_team=home_name,
                    away_team=away_name,
                    game_time=game.get("game_time"),
                    bet_type="Total",
                    side=side,
                    odds=under_odds,
                    model_prob=under_prob,
                    implied_prob=under_implied,
                    edge=under_edge,
                    kelly_capped=kelly,
                )
                picks.append(pick)

        return picks

    async def _save_and_format_pick(
        self,
        pick_date: date,
        espn_game_id: str,
        home_team: str,
        away_team: str,
        game_time: Optional[datetime],
        bet_type: str,
        side: str,
        odds: int,
        model_prob: float,
        implied_prob: float,
        edge: float,
        kelly_capped: float,
    ) -> dict:
        """Save a pick to the database and return formatted dict for API response."""
        db_pick = await self.picks_repo.create_pick(
            pick_date=pick_date,
            espn_game_id=espn_game_id,
            home_team=home_team,
            away_team=away_team,
            game_time=game_time,
            bet_type=bet_type,
            side=side,
            odds=odds,
            model_prob=Decimal(str(round(model_prob, 4))),
            implied_prob=Decimal(str(round(implied_prob, 4))),
            edge=Decimal(str(round(edge, 4))),
            kelly_capped=Decimal(str(round(kelly_capped, 4))),
        )

        return self._format_pick(db_pick)

    def _default_odds(self) -> dict:
        """Return default odds structure."""
        return {
            "moneyline": {"home_odds": -110, "away_odds": -110},
            "spread": {"line": 0, "home_odds": -110, "away_odds": -110},
            "total": {"line": 220, "over_odds": -110, "under_odds": -110},
        }

    def _format_pick(self, pick: DailyPick, game_status: str = "scheduled") -> dict:
        """Format a DailyPick database model to API response format."""
        game_time = pick.game_time
        if game_time:
            game_time_str = game_time.isoformat()
        else:
            game_time_str = f"{pick.pick_date}T19:00:00"

        return {
            "id": pick.id,
            "homeTeam": pick.home_team,
            "awayTeam": pick.away_team,
            "gameTime": game_time_str,
            "betType": pick.bet_type,
            "side": pick.side,
            "odds": pick.odds,
            "modelProb": float(pick.model_prob),
            "impliedProb": float(pick.implied_prob),
            "edge": float(pick.edge),
            "player": None,
            "gameStatus": game_status,
        }

    def _get_game_statuses(self, pick_date: date) -> Dict[str, str]:
        """
        Get current game statuses from ESPN for a date.
        Returns a dict mapping espn_game_id to status.
        """
        try:
            games = espn_data_service.get_games_for_date(pick_date)
            return {g["nba_game_id"]: g.get("status", "scheduled") for g in games}
        except Exception as e:
            logger.warning(f"Failed to fetch game statuses: {e}")
            return {}

    def _format_picks_with_status(
        self,
        picks: List[DailyPick],
        game_statuses: Dict[str, str]
    ) -> List[dict]:
        """
        Format picks with current game status.
        Filters out picks for completed games.
        """
        formatted_picks = []
        for pick in picks:
            status = game_statuses.get(pick.espn_game_id, "scheduled")

            # Skip picks for completed games
            if status == "final":
                continue

            formatted_picks.append(self._format_pick(pick, status))

        # Sort by edge descending
        formatted_picks.sort(key=lambda p: p.get("edge", 0), reverse=True)
        return formatted_picks
