"""API routes for picks endpoints."""

from datetime import date, datetime
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
    """
    picks_service = PicksService(db)

    # Parse date or use today
    if pick_date:
        try:
            target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()
        except ValueError:
            target_date = date.today()
    else:
        target_date = date.today()

    # Get picks
    picks = await picks_service.generate_picks_for_date(target_date)

    # Apply filters
    if bet_type:
        picks = [p for p in picks if p.get("betType", "").lower() == bet_type.lower()]

    if min_edge is not None:
        picks = [p for p in picks if p.get("edge", 0) >= min_edge]

    # Convert to response format
    pick_responses = [PickResponse(**p) for p in picks]

    return PicksResponse(picks=pick_responses)
