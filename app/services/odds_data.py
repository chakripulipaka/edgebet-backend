"""The Odds API integration service for fetching live betting odds."""

import logging
import time
from datetime import date
from typing import List, Optional, Tuple, Dict, Any
import httpx

from app.config import settings
from app.core.constants import NBA_TEAM_MAP

logger = logging.getLogger(__name__)

# Global cache for odds API with TTL
_odds_cache: Dict[str, Tuple[float, List[dict]]] = {}
_ODDS_CACHE_TTL = 300  # 5 minutes TTL for odds cache

# Explicit mapping from team abbreviations to The Odds API team names
ABBREV_TO_ODDS_API_NAME = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


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

    async def get_nba_odds(self, game_date: Optional[date] = None, force_refresh: bool = False) -> List[dict]:
        """
        Fetch NBA odds from The Odds API with caching.

        Args:
            game_date: Date to filter odds for (defaults to today)
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List of games with odds data
        """
        if not self.api_key:
            logger.warning("No Odds API key configured (THE_ODDS_API_KEY)")
            return []

        # Check cache first (unless force refresh)
        cache_key = str(game_date or "all")
        now = time.time()

        if not force_refresh and cache_key in _odds_cache:
            cached_time, cached_data = _odds_cache[cache_key]
            if now - cached_time < _ODDS_CACHE_TTL:
                logger.info(f"Returning cached odds ({len(cached_data)} games, age={(now - cached_time):.0f}s)")
                return cached_data

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

            logger.info(f"Fetched {len(events)} events from Odds API (fresh)")
            for event in events:
                logger.debug(f"  Odds API game: {event.get('away_team')} @ {event.get('home_team')}")

            # Note: We don't filter by date because:
            # 1. The API returns upcoming games (next 48 hours) automatically
            # 2. UTC timezone causes evening games to appear as next day
            # 3. match_game_to_odds() handles team matching anyway

            result = self._parse_odds_response(events)

            # Store in cache
            _odds_cache[cache_key] = (now, result)
            logger.info(f"Cached {len(result)} odds records for key '{cache_key}'")

            return result

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
            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                # Skip games without bookmaker data - no real odds available
                logger.debug(f"Skipping {event.get('away_team')} @ {event.get('home_team')} - no bookmakers")
                continue

            game = {
                "event_id": event.get("id"),
                "home_team": event.get("home_team"),  # Full name: "Boston Celtics"
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
                "odds": {},
            }

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
        so we use an explicit mapping for reliable matching.

        Args:
            home_team_abbrev: Home team abbreviation (e.g., "BOS")
            away_team_abbrev: Away team abbreviation (e.g., "BKN")
            odds_data: List of odds from get_nba_odds

        Returns:
            Matching odds dictionary or None
        """
        logger.info(f"Matching odds for {away_team_abbrev} @ {home_team_abbrev}")

        # Get expected Odds API names from our explicit mapping
        home_expected = ABBREV_TO_ODDS_API_NAME.get(home_team_abbrev)
        away_expected = ABBREV_TO_ODDS_API_NAME.get(away_team_abbrev)

        if not home_expected or not away_expected:
            logger.warning(
                f"Unknown team abbreviation: home={home_team_abbrev} ({home_expected}), "
                f"away={away_team_abbrev} ({away_expected})"
            )

        logger.debug(f"Looking for: '{away_expected}' @ '{home_expected}'")

        # Try exact name matching first (case-insensitive)
        for game in odds_data:
            game_home = game.get("home_team", "")
            game_away = game.get("away_team", "")

            if home_expected and away_expected:
                if game_home.lower() == home_expected.lower() and \
                   game_away.lower() == away_expected.lower():
                    odds = game.get("odds", {})
                    logger.info(
                        f"MATCHED {away_team_abbrev} @ {home_team_abbrev}: "
                        f"spread={odds.get('spread', {}).get('line')}, "
                        f"total={odds.get('total', {}).get('line')}"
                    )
                    return odds

        # Fallback to fuzzy matching (team name containment)
        home_info = self._get_team_info(home_team_abbrev)
        away_info = self._get_team_info(away_team_abbrev)

        for game in odds_data:
            game_home = game.get("home_team", "")
            game_away = game.get("away_team", "")

            if self._team_matches(home_info, game_home) and \
               self._team_matches(away_info, game_away):
                odds = game.get("odds", {})
                logger.info(
                    f"FUZZY MATCHED {away_team_abbrev} @ {home_team_abbrev} -> "
                    f"'{game_away}' @ '{game_home}': "
                    f"spread={odds.get('spread', {}).get('line')}, "
                    f"total={odds.get('total', {}).get('line')}"
                )
                return odds

        # No match found - log available games for debugging
        logger.warning(f"NO MATCH FOUND for {away_team_abbrev} @ {home_team_abbrev}")
        logger.warning(f"Available odds games ({len(odds_data)}):")
        for g in odds_data:
            logger.warning(f"  - {g.get('away_team')} @ {g.get('home_team')}")

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

        Note: We prioritize team name over city to avoid LA teams confusion
        (both Lakers and Clippers share "Los Angeles" city).
        """
        if not api_team_name:
            return False

        api_name_lower = api_team_name.lower()
        team_name = team_info.get("name", "").lower()

        # Team name MUST match - city alone is not sufficient
        # This prevents LAL/LAC confusion (both have city="Los Angeles")
        if not team_name:
            return False

        return team_name in api_name_lower


def clear_odds_cache():
    """Clear the odds cache. Called by hourly job to force fresh odds."""
    global _odds_cache
    _odds_cache = {}
    logger.info("Cleared odds cache")


# Singleton instance
odds_service = OddsDataService()
