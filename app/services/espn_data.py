"""ESPN API integration service for fetching NBA game data and stats."""

import logging
import time
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import httpx

from app.core.constants import ESPN_TO_NBA_TEAM_MAP, NBA_TO_ESPN_TEAM_MAP, NBA_TEAM_MAP

logger = logging.getLogger(__name__)


class ESPNDataService:
    """Service for fetching NBA data from ESPN's public API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    def __init__(self):
        self._request_delay = 0.5  # Rate limiting
        self._client = httpx.Client(timeout=30.0)

    def _delay(self):
        """Add delay between API requests to avoid rate limiting."""
        time.sleep(self._request_delay)

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make HTTP GET request to ESPN API."""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            self._delay()
            response = self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ESPN API request failed: {url} - {e}")
            return None

    def _espn_to_nba_team_id(self, espn_id: int) -> Optional[int]:
        """Convert ESPN team ID to NBA API team ID."""
        return ESPN_TO_NBA_TEAM_MAP.get(espn_id)

    def _nba_to_espn_team_id(self, nba_id: int) -> Optional[int]:
        """Convert NBA API team ID to ESPN team ID."""
        return NBA_TO_ESPN_TEAM_MAP.get(nba_id)

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
        date_str = game_date.strftime("%Y%m%d")
        data = self._make_request("/scoreboard", {"dates": date_str})

        if not data or "events" not in data:
            logger.info(f"No events found in ESPN response for {game_date}")
            return []

        games = []
        for event in data["events"]:
            game = self._parse_scoreboard_event(event, game_date)
            if game:
                games.append(game)

        logger.info(f"Parsed {len(games)} games from ESPN for {game_date}")
        return games

    def _parse_scoreboard_event(self, event: Dict, game_date: date) -> Optional[Dict]:
        """Parse a single event from scoreboard response."""
        try:
            event_id = event["id"]
            competitions = event.get("competitions", [])
            if not competitions:
                return None

            competition = competitions[0]
            competitors = competition.get("competitors", [])

            home_team = None
            away_team = None

            for comp in competitors:
                espn_team_id = int(comp["team"]["id"])
                nba_team_id = self._espn_to_nba_team_id(espn_team_id)
                if not nba_team_id:
                    logger.warning(f"Unknown ESPN team ID: {espn_team_id}")
                    continue

                score_str = comp.get("score", "0")
                score = int(score_str) if score_str and score_str.isdigit() else None

                if comp.get("homeAway") == "home":
                    home_team = {"id": nba_team_id, "score": score}
                else:
                    away_team = {"id": nba_team_id, "score": score}

            if not home_team or not away_team:
                return None

            status = self._parse_game_status(competition.get("status", {}))

            # Parse game time from ESPN (UTC format: "2026-02-08T02:00Z")
            game_time = None
            date_str = event.get("date")
            if date_str:
                try:
                    # Convert from UTC string to timezone-aware datetime
                    game_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse game time from '{date_str}': {e}")

            return {
                "nba_game_id": event_id,
                "game_date": game_date,
                "game_time": game_time,
                "home_team_id": home_team["id"],
                "away_team_id": away_team["id"],
                "status": status,
                "home_score": home_team["score"] if status == "final" else None,
                "away_score": away_team["score"] if status == "final" else None,
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse ESPN event: {e}")
            return None

    def _parse_game_status(self, status: Dict) -> str:
        """Convert ESPN status to our status string."""
        status_type = status.get("type", {})
        completed = status_type.get("completed", False)
        state = status_type.get("state", "")

        if completed:
            return "final"
        elif state == "in":
            return "in_progress"
        return "scheduled"

    def get_games_for_season(
        self, season: str, season_type: str = "Regular Season"
    ) -> List[dict]:
        """
        Fetch all games for a season.

        Args:
            season: Season string (e.g., "2024-25")
            season_type: "Regular Season" or "Playoffs"

        Returns:
            List of game dictionaries
        """
        # ESPN uses end year format (e.g., 2025 for 2024-25 season)
        year = int(season.split("-")[0]) + 1
        espn_season_type = 2 if season_type == "Regular Season" else 3

        all_games = {}

        # Fetch schedule for each team and aggregate
        for espn_id in ESPN_TO_NBA_TEAM_MAP.keys():
            data = self._make_request(
                f"/teams/{espn_id}/schedule",
                {"season": year, "seasontype": espn_season_type}
            )

            if not data or "events" not in data:
                continue

            for event in data["events"]:
                game_id = event["id"]
                if game_id in all_games:
                    continue

                game = self._parse_schedule_event(event, season)
                if game:
                    all_games[game_id] = game

            logger.info(f"Processed schedule for ESPN team {espn_id}, total games: {len(all_games)}")

        return list(all_games.values())

    def _parse_schedule_event(self, event: Dict, season: str) -> Optional[Dict]:
        """Parse a single event from team schedule response."""
        try:
            competitions = event.get("competitions", [])
            if not competitions:
                return None

            competition = competitions[0]
            competitors = competition.get("competitors", [])

            home_team = None
            away_team = None

            for comp in competitors:
                espn_team_id = int(comp["team"]["id"])
                nba_team_id = self._espn_to_nba_team_id(espn_team_id)
                if not nba_team_id:
                    continue

                # Score can be in different formats
                score_data = comp.get("score", {})
                if isinstance(score_data, dict):
                    score = int(score_data.get("value", 0)) if score_data.get("value") else None
                elif isinstance(score_data, str) and score_data.isdigit():
                    score = int(score_data)
                else:
                    score = None

                if comp.get("homeAway") == "home":
                    home_team = {"id": nba_team_id, "score": score}
                else:
                    away_team = {"id": nba_team_id, "score": score}

            if not home_team or not away_team:
                return None

            # Parse game time from ESPN (UTC format: "2026-02-08T02:00Z")
            game_time = None
            date_str = event.get("date", "")
            if date_str:
                try:
                    game_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    game_date = game_time.date()
                except (ValueError, AttributeError):
                    # Fallback to date-only parsing
                    game_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            else:
                game_date = None

            status = self._parse_game_status(competition.get("status", {}))

            return {
                "nba_game_id": event["id"],
                "season": season,
                "game_date": game_date,
                "game_time": game_time,
                "status": status,
                "home_team_id": home_team["id"],
                "away_team_id": away_team["id"],
                "home_score": home_team["score"] if status == "final" else None,
                "away_score": away_team["score"] if status == "final" else None,
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse ESPN schedule event: {e}")
            return None

    def get_box_score(self, game_id: str) -> Dict[int, Dict]:
        """
        Fetch detailed box score for a game.

        Args:
            game_id: ESPN event ID

        Returns:
            Dictionary with team stats keyed by NBA team ID
        """
        data = self._make_request("/summary", {"event": game_id})

        if not data:
            logger.info(f"No data returned from ESPN summary for game {game_id}")
            return {}

        # Check if boxscore exists in response
        if "boxscore" not in data:
            logger.info(f"No boxscore in ESPN summary for game {game_id}")
            return {}

        result = {}
        boxscore = data["boxscore"]
        teams = boxscore.get("teams", [])

        # Extract scores from header (more reliable than boxscore stats)
        team_scores = {}
        header = data.get("header", {})
        competitions = header.get("competitions", [])
        if competitions:
            for comp in competitions[0].get("competitors", []):
                espn_id = int(comp.get("team", {}).get("id", 0))
                score_str = comp.get("score", "")
                if score_str and score_str.isdigit():
                    team_scores[espn_id] = int(score_str)

        logger.info(f"ESPN boxscore has {len(teams)} teams for game {game_id}, scores: {team_scores}")

        for team_data in teams:
            team_info = team_data.get("team", {})
            espn_team_id = int(team_info.get("id", 0))
            nba_team_id = self._espn_to_nba_team_id(espn_team_id)

            if not nba_team_id:
                logger.warning(f"Unknown ESPN team ID in boxscore: {espn_team_id}")
                continue

            stats = self._parse_team_box_score_stats(team_data.get("statistics", []))
            # Add score from header
            stats["points"] = team_scores.get(espn_team_id)
            result[nba_team_id] = stats
            logger.info(f"Parsed stats for team {nba_team_id}: {stats}")

        return result

    def _parse_team_box_score_stats(self, statistics: List[Dict]) -> Dict:
        """Parse team statistics from box score."""
        stats = {
            "points": None,
            "fg_pct": None,
            "fg3_pct": None,
            "ft_pct": None,
            "assists": None,
            "rebounds": None,
            "blocks": None,
            "steals": None,
            "turnovers": None,
        }

        # ESPN stat name to our field name mapping
        stat_mapping = {
            "points": "points",
            "fieldGoalPct": "fg_pct",
            "threePointFieldGoalPct": "fg3_pct",
            "freeThrowPct": "ft_pct",
            "assists": "assists",
            "totalRebounds": "rebounds",
            "rebounds": "rebounds",
            "blocks": "blocks",
            "steals": "steals",
            "turnovers": "turnovers",
        }

        for stat_obj in statistics:
            name = stat_obj.get("name", "")
            display_value = stat_obj.get("displayValue", "")

            if name in stat_mapping:
                key = stat_mapping[name]
                try:
                    if key in ["fg_pct", "fg3_pct", "ft_pct"]:
                        # Percentages come as "45.2" format, convert to decimal
                        val = float(display_value)
                        stats[key] = val / 100.0 if val > 1 else val
                    else:
                        stats[key] = int(float(display_value))
                except (ValueError, TypeError):
                    pass

        return stats

    def get_game_stats_fallback(self, game_id: str) -> Dict[int, Dict]:
        """
        Fallback method - ESPN summary endpoint usually has all data.
        Just calls get_box_score.
        """
        return self.get_box_score(game_id)

    def get_team_game_logs(
        self, team_id: int, season: str, num_games: int = 10
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
        espn_team_id = self._nba_to_espn_team_id(team_id)
        if not espn_team_id:
            logger.warning(f"No ESPN team ID found for NBA team {team_id}")
            return []

        year = int(season.split("-")[0]) + 1

        data = self._make_request(
            f"/teams/{espn_team_id}/schedule",
            {"season": year, "seasontype": 2}
        )

        if not data or "events" not in data:
            return []

        # Filter completed games and sort by date descending
        completed_games = []
        for event in data["events"]:
            competitions = event.get("competitions", [])
            if not competitions:
                continue

            status = self._parse_game_status(competitions[0].get("status", {}))
            if status != "final":
                continue

            completed_games.append(event)

        # Sort by date descending and take top N
        completed_games.sort(
            key=lambda x: x.get("date", ""),
            reverse=True
        )
        completed_games = completed_games[:num_games]

        stats = []
        for event in completed_games:
            game_log = self._parse_team_game_log(event, team_id)
            if game_log:
                stats.append(game_log)

        return stats

    def _parse_team_game_log(self, event: Dict, nba_team_id: int) -> Optional[Dict]:
        """Parse game log for a specific team."""
        try:
            competitions = event.get("competitions", [])
            if not competitions:
                return None

            competition = competitions[0]
            competitors = competition.get("competitors", [])

            target_team = None
            is_home = False

            for comp in competitors:
                espn_id = int(comp["team"]["id"])
                if self._espn_to_nba_team_id(espn_id) == nba_team_id:
                    target_team = comp
                    is_home = comp.get("homeAway") == "home"
                    break

            if not target_team:
                return None

            date_str = event.get("date", "")[:10]
            game_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Get score
            score_data = target_team.get("score", {})
            if isinstance(score_data, dict):
                points = int(score_data.get("value", 0)) if score_data.get("value") else None
            elif isinstance(score_data, str) and score_data.isdigit():
                points = int(score_data)
            else:
                points = None

            return {
                "game_id": event["id"],
                "game_date": game_date,
                "is_home": is_home,
                "points": points,
                "fg_pct": None,
                "fg3_pct": None,
                "ft_pct": None,
                "assists": None,
                "rebounds": None,
                "blocks": None,
                "steals": None,
                "turnovers": None,
                "plus_minus": None,
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse ESPN game log: {e}")
            return None

    def get_game_result(self, game_id: str) -> Optional[Dict]:
        """
        Fetch final scores for a completed game.

        Args:
            game_id: ESPN event ID

        Returns:
            Dictionary with home_score, away_score, home_team, away_team, status
            or None if game not found/not completed
        """
        data = self._make_request("/summary", {"event": game_id})

        if not data:
            return None

        header = data.get("header", {})
        competitions = header.get("competitions", [])

        if not competitions:
            return None

        competition = competitions[0]
        competitors = competition.get("competitors", [])

        result = {
            "home_score": None,
            "away_score": None,
            "home_team": None,
            "away_team": None,
            "status": "unknown",
        }

        for comp in competitors:
            team_name = comp.get("team", {}).get("displayName", "")
            score_str = comp.get("score", "")
            score = int(score_str) if score_str and score_str.isdigit() else None

            if comp.get("homeAway") == "home":
                result["home_team"] = team_name
                result["home_score"] = score
            else:
                result["away_team"] = team_name
                result["away_score"] = score

        # Check game status
        status_data = competition.get("status", {})
        status_type = status_data.get("type", {})
        if status_type.get("completed", False):
            result["status"] = "final"
        elif status_type.get("state") == "in":
            result["status"] = "in_progress"
        else:
            result["status"] = "scheduled"

        return result

    def calculate_rest_days(
        self, team_id: int, game_date: date, previous_games: List[dict]
    ) -> int:
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

        previous_dates = [g["game_date"] for g in previous_games if g["game_date"] < game_date]
        if not previous_dates:
            return 3

        last_game = max(previous_dates)
        rest_days = (game_date - last_game).days - 1
        return max(0, rest_days)


# Singleton instance
espn_data_service = ESPNDataService()
