"""API routes for picks endpoints."""

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.repositories.picks import PicksRepository
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

    # Get picks for target date
    picks = await picks_service.generate_picks_for_date(target_date)
    all_games_complete = picks_service.check_all_games_complete(target_date)

    # Apply filters
    if bet_type:
        picks = [p for p in picks if p.get("betType", "").lower() == bet_type.lower()]

    if min_edge is not None:
        picks = [p for p in picks if p.get("edge", 0) >= min_edge]

    # Convert to response format
    pick_responses = [PickResponse(**p) for p in picks]

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
