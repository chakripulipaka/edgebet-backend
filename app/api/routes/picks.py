"""API routes for picks endpoints."""

from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

# Use US Eastern timezone for "today" since NBA games are scheduled in ET
EASTERN_TZ = ZoneInfo("America/New_York")

from app.db.database import get_db
from app.db.repositories.picks import PicksRepository
from app.api.schemas.pick import PicksResponse, PickResponse
from app.services.picks_service import PicksService

router = APIRouter()


def parse_game_time(game_time_str: str) -> datetime:
    """Parse game time string to datetime, handling timezone."""
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
        # Convert to Eastern if it has timezone info
        if dt.tzinfo:
            return dt.astimezone(EASTERN_TZ)
        # Assume Eastern if no timezone
        return dt.replace(tzinfo=EASTERN_TZ)
    except (ValueError, AttributeError):
        # Fallback: return a far-future date so it doesn't get filtered out
        return datetime.now(EASTERN_TZ) + timedelta(days=365)


@router.get("/picks", response_model=PicksResponse)
async def get_picks(
    hours: int = Query(36, description="Hours ahead to fetch picks for (default 36)"),
    pick_date: Optional[str] = Query(None, description="Specific date (overrides hours param)"),
    bet_type: Optional[str] = Query(None, description="Filter by bet type"),
    min_edge: Optional[float] = Query(None, description="Minimum edge percentage"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get upcoming +EV betting picks for the next N hours.

    By default, returns picks for games in the next 36 hours.
    If pick_date is specified, returns picks for that specific date only.

    Returns picks sorted by game time, with live games indicated by gameStatus.
    """
    picks_service = PicksService(db)

    # If specific date requested, use old behavior
    if pick_date:
        try:
            target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()
            picks = await picks_service.generate_picks_for_date(target_date)
            all_games_complete = picks_service.check_all_games_complete(target_date)

            if bet_type:
                picks = [p for p in picks if p.get("betType", "").lower() == bet_type.lower()]
            if min_edge is not None:
                picks = [p for p in picks if p.get("edge", 0) >= min_edge]

            pick_responses = [PickResponse(**p) for p in picks]
            return PicksResponse(picks=pick_responses, allGamesComplete=all_games_complete)
        except ValueError:
            pass  # Fall through to rolling window behavior

    # Rolling window: get picks for games in the next N hours
    now = datetime.now(EASTERN_TZ)
    end_time = now + timedelta(hours=hours)

    # Determine which dates we need to query
    start_date = now.date()
    end_date = end_time.date()

    # Collect picks from all relevant dates
    all_picks = []
    dates_to_check = [start_date]
    if end_date != start_date:
        dates_to_check.append(end_date)

    for current_date in dates_to_check:
        date_picks = await picks_service.generate_picks_for_date(current_date)
        all_picks.extend(date_picks)

    # Filter to games within the time window AND not final
    # (Frontend will also filter, but we can reduce payload)
    filtered_picks = []
    for pick in all_picks:
        game_time = parse_game_time(pick.get("gameTime", ""))
        game_status = pick.get("gameStatus", "scheduled")

        # Include if:
        # - Game is live (in_progress or halftime) - always show
        # - Game is scheduled and within the time window
        if game_status in ("in_progress", "halftime"):
            filtered_picks.append(pick)
        elif game_status == "scheduled" and game_time <= end_time:
            filtered_picks.append(pick)
        # Skip final games - they should disappear

    # Apply additional filters
    if bet_type:
        filtered_picks = [p for p in filtered_picks if p.get("betType", "").lower() == bet_type.lower()]

    if min_edge is not None:
        filtered_picks = [p for p in filtered_picks if p.get("edge", 0) >= min_edge]

    # Sort: live games first, then by game time
    def sort_key(pick):
        status = pick.get("gameStatus", "scheduled")
        is_live = status in ("in_progress", "halftime")
        game_time = parse_game_time(pick.get("gameTime", ""))
        # Live games get priority (0), scheduled get (1), so live comes first
        return (0 if is_live else 1, game_time)

    filtered_picks.sort(key=sort_key)

    # Convert to response format
    pick_responses = [PickResponse(**p) for p in filtered_picks]

    # All games complete only if no picks remain (all were final or none scheduled)
    all_games_complete = len(pick_responses) == 0

    return PicksResponse(picks=pick_responses, allGamesComplete=all_games_complete)


@router.delete("/picks/{pick_date}")
async def delete_picks_for_date(
    pick_date: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete all picks for a specific date."""
    try:
        target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD", "deleted": 0}

    picks_repo = PicksRepository(db)
    deleted_count = await picks_repo.delete_all_for_date(target_date)
    return {"deleted": deleted_count, "date": pick_date}
