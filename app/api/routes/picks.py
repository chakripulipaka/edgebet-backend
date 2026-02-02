"""API routes for picks endpoints."""

from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Use US Eastern timezone for "today" since NBA games are scheduled in ET
EASTERN_TZ = ZoneInfo("America/New_York")

from app.db.database import get_db
from app.db.models import PicksSnapshot
from app.db.repositories.picks import PicksRepository
from app.api.schemas.pick import PicksResponse, PickResponse, GameWithoutPicks
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

    Returns pre-computed picks from hourly snapshot for instant loading.
    Falls back to live computation if no snapshot exists yet.

    If pick_date is specified, returns picks for that specific date only.
    """
    # If specific date requested, use live computation
    if pick_date:
        try:
            target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()
            picks_service = PicksService(db)
            picks = await picks_service.generate_picks_for_date(target_date)

            if bet_type:
                picks = [p for p in picks if p.get("betType", "").lower() == bet_type.lower()]
            if min_edge is not None:
                picks = [p for p in picks if p.get("edge", 0) >= min_edge]

            # Check if all games are complete based on gameStatus in formatted picks
            all_games_complete = (
                len(picks) > 0 and
                all(p.get("gameStatus") == "final" for p in picks)
            )

            pick_responses = [PickResponse(**p) for p in picks]
            return PicksResponse(picks=pick_responses, allGamesComplete=all_games_complete)
        except ValueError:
            pass  # Fall through to snapshot behavior

    # Try to get the latest snapshot (fast path - instant response)
    result = await db.execute(
        select(PicksSnapshot)
        .order_by(PicksSnapshot.computed_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if snapshot:
        # Return pre-computed data from snapshot
        snapshot_data = snapshot.snapshot_data
        picks_data = snapshot_data.get("picks", [])
        games_without_picks = snapshot_data.get("gamesWithoutPicks", [])

        # Apply optional filters
        if bet_type:
            picks_data = [p for p in picks_data if p.get("betType", "").lower() == bet_type.lower()]
        if min_edge is not None:
            picks_data = [p for p in picks_data if p.get("edge", 0) >= min_edge]

        pick_responses = [PickResponse(**p) for p in picks_data]
        games_without_responses = [GameWithoutPicks(**g) for g in games_without_picks]

        return PicksResponse(
            picks=pick_responses,
            gamesWithoutPicks=games_without_responses,
            allGamesComplete=snapshot_data.get("allGamesComplete", False),
            computedAt=snapshot_data.get("computedAt"),
        )

    # No snapshot yet - fall back to live computation
    # This only happens on first load before any hourly job has run
    picks_service = PicksService(db)

    now = datetime.now(EASTERN_TZ)
    end_time = now + timedelta(hours=hours)

    # Determine which dates we need to query
    start_date = now.date()
    end_date = end_time.date()

    # Collect picks from ALL dates in the window (not just start/end)
    all_picks = []
    current_date = start_date
    while current_date <= end_date:
        date_picks = await picks_service.generate_picks_for_date(current_date)
        all_picks.extend(date_picks)
        current_date += timedelta(days=1)

    # Filter to games within the time window AND not final
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
        return (0 if is_live else 1, game_time)

    filtered_picks.sort(key=sort_key)

    # Convert to response format
    pick_responses = [PickResponse(**p) for p in filtered_picks]

    # All games complete only if no picks remain
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
