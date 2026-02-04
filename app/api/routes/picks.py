"""API routes for picks endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import PicksSnapshot
from app.db.repositories.picks import PicksRepository
from app.api.schemas.pick import PicksResponse, PickResponse, GameWithoutPicks

router = APIRouter()


@router.get("/picks", response_model=PicksResponse)
async def get_picks(
    bet_type: Optional[str] = Query(None, description="Filter by bet type"),
    min_edge: Optional[float] = Query(None, description="Minimum edge percentage"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get upcoming +EV betting picks.

    Returns pre-computed picks from the latest hourly snapshot.
    All computation happens in the background hourly job - this
    endpoint only reads from the database for instant response.
    """
    # Get the latest pre-computed snapshot from the database.
    # The hourly job computes all picks in the background - users only
    # read from the snapshot and never trigger live computation.
    result = await db.execute(
        select(PicksSnapshot)
        .order_by(PicksSnapshot.computed_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot:
        # No snapshot yet - return empty data (hourly job hasn't run yet)
        return PicksResponse(
            picks=[],
            gamesWithoutPicks=[],
            allGamesComplete=False,
            computedAt=None,
        )

    # Return pre-computed data from snapshot (instant)
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
