"""The Odds API integration service for fetching live betting odds."""

import logging
from datetime import date
from typing import List, Optional
import httpx

from app.config import settings
from app.core.constants import NBA_TEAM_MAP

logger = logging.getLogger(__name__)


class OddsDataService:
    """Service for fetching odds from The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self):
        self.api_key = settings.THE_ODDS_API_KEY
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_nba_odds(self, game_date: Optional[date] = None) -> List[dict]:
        """
        Fetch NBA odds from The Odds API.

        Args:
            game_date: Date to filter odds for (defaults to today)

        Returns:
            List of games with odds data
        """
        if not self.api_key:
            logger.warning("No Odds API key configured (THE_ODDS_API_KEY)")
            return []

        try:
            client = await self._get_client()

            response = await client.get(
                f"{self.BASE_URL}/sports/basketball_nba/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": "spreads,totals,h2h",
                    "oddsFormat": "american",
                },
            )
            response.raise_for_status()

            events = response.json()  # Direct array, not wrapped

            # Note: We don't filter by date because:
            # 1. The API returns upcoming games (next 48 hours) automatically
            # 2. UTC timezone causes evening games to appear as next day
            # 3. match_game_to_odds() handles team matching anyway

            return self._parse_odds_response(events)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Invalid Odds API key")
            else:
                logger.error(f"Odds API HTTP error: {e.response.status_code}")
            return []
        except httpx.HTTPError as e:
            logger.error(f"Odds API network error: {e}")
            return []
        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return []

    def _parse_odds_response(self, events: List[dict]) -> List[dict]:
        """Parse The Odds API response into our odds format."""
        games = []

        for event in events:
            game = {
                "event_id": event.get("id"),
                "home_team": event.get("home_team"),  # Full name: "Boston Celtics"
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
                "odds": {},
            }

            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                games.append(game)
                continue

            # Use first available bookmaker (FanDuel, DraftKings, etc.)
            bookmaker = bookmakers[0]
            home_team = event.get("home_team")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                if market_key == "h2h":
                    game["odds"]["moneyline"] = self._parse_h2h_market(
                        market, home_team
                    )
                elif market_key == "spreads":
                    game["odds"]["spread"] = self._parse_spread_market(
                        market, home_team
                    )
                elif market_key == "totals":
                    game["odds"]["total"] = self._parse_totals_market(market)

            games.append(game)

        return games

    def _parse_h2h_market(self, market: dict, home_team: str) -> dict:
        """Parse head-to-head (moneyline) market."""
        result = {"home_odds": -110, "away_odds": -110}

        for outcome in market.get("outcomes", []):
            team = outcome.get("name")
            # The Odds API returns American odds directly
            odds = outcome.get("price", -110)

            if team == home_team:
                result["home_odds"] = odds
            else:
                result["away_odds"] = odds

        return result

    def _parse_spread_market(self, market: dict, home_team: str) -> dict:
        """Parse spread market."""
        result = {"line": 0, "home_odds": -110, "away_odds": -110}

        for outcome in market.get("outcomes", []):
            team = outcome.get("name")
            point = outcome.get("point", 0)
            odds = outcome.get("price", -110)

            if team == home_team:
                result["line"] = point
                result["home_odds"] = odds
            else:
                result["away_odds"] = odds

        return result

    def _parse_totals_market(self, market: dict) -> dict:
        """Parse totals (over/under) market."""
        result = {"line": 220, "over_odds": -110, "under_odds": -110}

        for outcome in market.get("outcomes", []):
            name = outcome.get("name", "").lower()
            point = outcome.get("point", 220)
            odds = outcome.get("price", -110)

            result["line"] = point
            if name == "over":
                result["over_odds"] = odds
            else:
                result["under_odds"] = odds

        return result

    def match_game_to_odds(
        self,
        home_team_abbrev: str,
        away_team_abbrev: str,
        odds_data: List[dict],
    ) -> Optional[dict]:
        """
        Match a game to its odds data using team abbreviations.

        The Odds API returns full team names (e.g., "Boston Celtics"),
        so we match against team name and city from our mapping.

        Args:
            home_team_abbrev: Home team abbreviation (e.g., "BOS")
            away_team_abbrev: Away team abbreviation (e.g., "BKN")
            odds_data: List of odds from get_nba_odds

        Returns:
            Matching odds dictionary or None
        """
        # Get team info from our mapping
        home_info = self._get_team_info(home_team_abbrev)
        away_info = self._get_team_info(away_team_abbrev)

        for game in odds_data:
            game_home = game.get("home_team", "")
            game_away = game.get("away_team", "")

            # Match using full name, team name, or city
            if self._team_matches(home_info, game_home) and \
               self._team_matches(away_info, game_away):
                return game.get("odds", {})

        return None

    def _get_team_info(self, abbrev: str) -> dict:
        """Get team info from abbreviation."""
        for team_id, info in NBA_TEAM_MAP.items():
            if info.get("abbreviation") == abbrev:
                return info
        return {"abbreviation": abbrev, "name": abbrev, "city": ""}

    def _team_matches(self, team_info: dict, api_team_name: str) -> bool:
        """
        Check if our team info matches the API team name.

        API returns names like "Boston Celtics", so we check if:
        - Team name is in API name ("Celtics" in "Boston Celtics")
        - City is in API name ("Boston" in "Boston Celtics")
        """
        if not api_team_name:
            return False

        api_name_lower = api_team_name.lower()
        team_name = team_info.get("name", "").lower()
        city = team_info.get("city", "").lower()

        return (team_name and team_name in api_name_lower) or \
               (city and city in api_name_lower)


# Singleton instance
odds_service = OddsDataService()
