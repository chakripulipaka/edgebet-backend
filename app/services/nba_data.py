"""NBA API integration service for fetching game data and stats."""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional
import time

from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoresummaryv2,
    scoreboardv2,
)
from nba_api.stats.static import teams as nba_teams

from app.core.constants import NBA_TEAM_MAP

logger = logging.getLogger(__name__)


class NBADataService:
    """Service for fetching NBA data from the official NBA API."""

    def __init__(self):
        self._request_delay = 0.6  # Rate limiting

    def _delay(self):
        """Add delay between API requests to avoid rate limiting."""
        time.sleep(self._request_delay)

    def get_all_teams(self) -> List[dict]:
        """
        Get all NBA teams with their IDs and info.

        Returns:
            List of team dictionaries with nba_team_id, abbreviation, name, city
        """
        teams = []
        for team_id, info in NBA_TEAM_MAP.items():
            teams.append({
                "nba_team_id": team_id,
                "abbreviation": info["abbreviation"],
                "name": info["name"],
                "city": info["city"],
            })
        return teams

    def get_games_for_date(self, game_date: date) -> List[dict]:
        """
        Fetch all games scheduled for a specific date.

        Args:
            game_date: The date to fetch games for

        Returns:
            List of game dictionaries
        """
        self._delay()
        try:
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=game_date.strftime("%Y-%m-%d")
            )
            games_data = scoreboard.game_header.get_data_frame()

            games = []
            for _, row in games_data.iterrows():
                game_time = None
                if row.get("GAME_STATUS_TEXT"):
                    # Try to parse game time
                    try:
                        time_str = row.get("GAME_STATUS_TEXT", "")
                        if "ET" in time_str or "PM" in time_str or "AM" in time_str:
                            # Parse time string
                            pass
                    except Exception:
                        pass

                # Convert status_id to int for reliable comparison (pandas returns numpy int64)
                status_id = int(row.get("GAME_STATUS_ID", 1))

                games.append({
                    "nba_game_id": str(row["GAME_ID"]),
                    "game_date": game_date,
                    "game_time": game_time,
                    "home_team_id": int(row["HOME_TEAM_ID"]),
                    "away_team_id": int(row["VISITOR_TEAM_ID"]),
                    "status": self._parse_status(status_id),
                    "home_score": int(row["HOME_TEAM_SCORE"]) if status_id == 3 and row.get("HOME_TEAM_SCORE") else None,
                    "away_score": int(row["VISITOR_TEAM_SCORE"]) if status_id == 3 and row.get("VISITOR_TEAM_SCORE") else None,
                })
            return games
        except Exception as e:
            logger.error(f"Error fetching games for {game_date}: {e}")
            return []

    def _parse_status(self, status_id: int) -> str:
        """Convert NBA API status ID to our status string."""
        status_map = {
            1: "scheduled",
            2: "in_progress",
            3: "final",
        }
        return status_map.get(status_id, "scheduled")

    def get_games_for_season(
        self,
        season: str,
        season_type: str = "Regular Season"
    ) -> List[dict]:
        """
        Fetch all games for a season.

        Args:
            season: Season string (e.g., "2024-25")
            season_type: "Regular Season" or "Playoffs"

        Returns:
            List of game dictionaries
        """
        self._delay()
        try:
            # Use LeagueGameFinder to get all games
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",  # NBA
            )
            games_df = game_finder.get_data_frames()[0]

            # Group by game ID to get both teams
            game_map = {}
            for _, row in games_df.iterrows():
                game_id = str(row["GAME_ID"])
                team_id = int(row["TEAM_ID"])
                matchup = row["MATCHUP"]
                is_home = "@" not in matchup

                if game_id not in game_map:
                    game_map[game_id] = {
                        "nba_game_id": game_id,
                        "season": season,
                        "game_date": datetime.strptime(row["GAME_DATE"], "%Y-%m-%d").date(),
                        "status": "final" if row.get("WL") else "scheduled",
                    }

                if is_home:
                    game_map[game_id]["home_team_id"] = team_id
                    game_map[game_id]["home_score"] = int(row["PTS"]) if row.get("PTS") else None
                else:
                    game_map[game_id]["away_team_id"] = team_id
                    game_map[game_id]["away_score"] = int(row["PTS"]) if row.get("PTS") else None

            # Filter out games missing teams
            return [g for g in game_map.values()
                    if g.get("home_team_id") and g.get("away_team_id")]
        except Exception as e:
            logger.error(f"Error fetching games for season {season}: {e}")
            return []

    def get_box_score(self, game_id: str) -> dict:
        """
        Fetch detailed box score for a game.

        Args:
            game_id: NBA API game ID

        Returns:
            Dictionary with team stats
        """
        self._delay()
        try:
            box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            team_stats_df = box_score.team_stats.get_data_frame()

            logger.info(f"Box score DataFrame for {game_id}: {len(team_stats_df)} rows")
            if team_stats_df.empty:
                logger.info(f"No team stats data in box score for game {game_id}")
                return {}

            result = {}
            for _, row in team_stats_df.iterrows():
                team_id = int(row["TEAM_ID"])
                result[team_id] = {
                    "points": int(row["PTS"]) if row.get("PTS") else None,
                    "fg_pct": float(row["FG_PCT"]) if row.get("FG_PCT") else None,
                    "fg3_pct": float(row["FG3_PCT"]) if row.get("FG3_PCT") else None,
                    "ft_pct": float(row["FT_PCT"]) if row.get("FT_PCT") else None,
                    "assists": int(row["AST"]) if row.get("AST") else None,
                    "rebounds": int(row["REB"]) if row.get("REB") else None,
                    "blocks": int(row["BLK"]) if row.get("BLK") else None,
                    "steals": int(row["STL"]) if row.get("STL") else None,
                    "turnovers": int(row["TO"]) if row.get("TO") else None,
                }
            return result
        except Exception as e:
            logger.error(f"Error fetching box score for game {game_id}: {e}")
            return {}

    def get_game_stats_fallback(self, game_id: str) -> dict:
        """
        Fallback method to get game stats using BoxScoreSummaryV2.

        This endpoint is more reliable for getting basic game scores
        when boxscoretraditionalv2 returns empty.

        Args:
            game_id: NBA API game ID

        Returns:
            Dictionary with team stats keyed by team_id
        """
        self._delay()
        try:
            # Try BoxScoreSummaryV2 - more reliable for game scores
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
            line_score_df = summary.line_score.get_data_frame()

            if line_score_df.empty:
                logger.info(f"No line score data in summary for game {game_id}")
                return {}

            logger.info(f"Found {len(line_score_df)} rows for game {game_id} via BoxScoreSummaryV2")

            result = {}
            for _, row in line_score_df.iterrows():
                team_id = int(row["TEAM_ID"])
                # Line score has PTS column for final score
                pts = row.get("PTS")
                result[team_id] = {
                    "points": int(pts) if pts is not None else None,
                    # Line score doesn't have detailed stats, just scores
                    # These will be None but at least we get the scores
                    "fg_pct": float(row["FG_PCT"]) if row.get("FG_PCT") else None,
                    "fg3_pct": float(row["FG3_PCT"]) if row.get("FG3_PCT") else None,
                    "ft_pct": float(row["FT_PCT"]) if row.get("FT_PCT") else None,
                    "assists": int(row["AST"]) if row.get("AST") else None,
                    "rebounds": int(row["REB"]) if row.get("REB") else None,
                    "blocks": None,  # Not in line score
                    "steals": None,  # Not in line score
                    "turnovers": int(row["TOV"]) if row.get("TOV") else None,
                }
            return result
        except Exception as e:
            logger.error(f"Error in fallback stats fetch for game {game_id}: {e}")
            return {}

    def get_team_game_logs(
        self,
        team_id: int,
        season: str,
        num_games: int = 10
    ) -> List[dict]:
        """
        Get recent game logs for a team.

        Args:
            team_id: NBA team ID
            season: Season string
            num_games: Number of recent games to fetch

        Returns:
            List of game stat dictionaries
        """
        self._delay()
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            games_df = game_finder.get_data_frames()[0]

            if games_df.empty:
                return []

            # Sort by date descending and take top N
            games_df = games_df.sort_values("GAME_DATE", ascending=False).head(num_games)

            stats = []
            for _, row in games_df.iterrows():
                matchup = row["MATCHUP"]
                is_home = "@" not in matchup

                stats.append({
                    "game_id": str(row["GAME_ID"]),
                    "game_date": datetime.strptime(row["GAME_DATE"], "%Y-%m-%d").date(),
                    "is_home": is_home,
                    "points": int(row["PTS"]) if row.get("PTS") else None,
                    "fg_pct": float(row["FG_PCT"]) if row.get("FG_PCT") else None,
                    "fg3_pct": float(row["FG3_PCT"]) if row.get("FG3_PCT") else None,
                    "ft_pct": float(row["FT_PCT"]) if row.get("FT_PCT") else None,
                    "assists": int(row["AST"]) if row.get("AST") else None,
                    "rebounds": int(row["REB"]) if row.get("REB") else None,
                    "blocks": int(row["BLK"]) if row.get("BLK") else None,
                    "steals": int(row["STL"]) if row.get("STL") else None,
                    "turnovers": int(row["TO"]) if row.get("TO") else None,
                    "plus_minus": int(row["PLUS_MINUS"]) if row.get("PLUS_MINUS") else None,
                })
            return stats
        except Exception as e:
            logger.error(f"Error fetching game logs for team {team_id}: {e}")
            return []

    def calculate_rest_days(self, team_id: int, game_date: date, previous_games: List[dict]) -> int:
        """
        Calculate rest days since last game.

        Args:
            team_id: NBA team ID
            game_date: Current game date
            previous_games: List of previous game dictionaries with game_date

        Returns:
            Number of rest days (0 = back-to-back)
        """
        if not previous_games:
            return 3  # Default if no history

        # Find most recent game before this date
        previous_dates = [g["game_date"] for g in previous_games if g["game_date"] < game_date]
        if not previous_dates:
            return 3

        last_game = max(previous_dates)
        rest_days = (game_date - last_game).days - 1
        return max(0, rest_days)


# Singleton instance
nba_data_service = NBADataService()
