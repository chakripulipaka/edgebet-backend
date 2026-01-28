"""API routes for picks endpoints."""

from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.api.schemas.pick import PicksResponse, PickResponse
from app.services.picks_service import PicksService

router = APIRouter()


@router.get("/picks", response_model=PicksResponse)
async def get_picks(
    pick_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    bet_type: Optional[str] = Query(None, description="Filter by bet type"),
    min_edge: Optional[float] = Query(None, description="Minimum edge percentage"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get today's top +EV betting picks.

    Returns picks sorted by edge, optionally filtered by bet type and minimum edge.
    Includes allGamesComplete flag to indicate when all games for the day are final.

    Auto-transitions to tomorrow's picks when today is complete (unless specific date requested).
    """
    picks_service = PicksService(db)

    # Parse date or use today
    specific_date_requested = pick_date is not None
    if pick_date:
        try:
            target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()
        except ValueError:
            target_date = date.today()
            specific_date_requested = False
    else:
        target_date = date.today()

    # Check if all games for target date are complete
    all_games_complete = picks_service.check_all_games_complete(target_date)

    # Auto-transition: If today is complete and no specific date was requested,
    # automatically show tomorrow's picks
    if all_games_complete and not specific_date_requested:
        tomorrow = target_date + timedelta(days=1)
        # Generate picks for tomorrow (will use cached if already generated)
        picks = await picks_service.generate_picks_for_date(tomorrow)
        # Tomorrow's games won't be complete yet
        all_games_complete = picks_service.check_all_games_complete(tomorrow)
    else:
        picks = await picks_service.generate_picks_for_date(target_date)

    # Apply filters
    if bet_type:
        picks = [p for p in picks if p.get("betType", "").lower() == bet_type.lower()]

    if min_edge is not None:
        picks = [p for p in picks if p.get("edge", 0) >= min_edge]

    # Convert to response format
    pick_responses = [PickResponse(**p) for p in picks]

    return PicksResponse(picks=pick_responses, allGamesComplete=all_games_complete)
