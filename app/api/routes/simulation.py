"""API routes for simulation endpoints."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.repositories.picks import PicksRepository
from app.api.schemas.simulation import SimulationResponse
from app.services.simulation_service import SimulationService
from app.services.picks_service import PicksService
from app.jobs.daily_job import resolve_and_cleanup_picks, resolve_all_picks, run_daily_job

router = APIRouter()


@router.get("/simulation", response_model=SimulationResponse)
async def get_simulation(
    starting_bankroll: float = Query(100.0, description="Starting portfolio value"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get simulation data based on stored picks.

    Runs simulation dynamically by:
    1. Getting all stored picks from the database
    2. For each day with picks, selecting the highest-edge pick
    3. Fetching game results from ESPN
    4. Calculating outcomes and updating bankroll with Kelly sizing

    Returns bankroll progression, win/loss record, and bet history.
    """
    sim_service = SimulationService(db)
    data = await sim_service.run_live_simulation(starting_bankroll)

    return SimulationResponse(**data)


@router.post("/simulation/backfill")
async def backfill_simulation(db: AsyncSession = Depends(get_db)):
    """
    Backfill outcomes for picks that don't have them stored.
    Fetches game results from ESPN and stores outcomes.
    """
    picks_repo = PicksRepository(db)
    pending_picks = await picks_repo.get_pending_picks()

    if not pending_picks:
        return {"message": "No pending picks to backfill", "resolved": 0}

    # Get unique dates
    dates = sorted(set(p.pick_date for p in pending_picks))
    resolved_count = 0

    for pick_date in dates:
        await resolve_and_cleanup_picks(db, pick_date)
        resolved_count += 1

    return {"message": f"Backfilled {resolved_count} dates", "resolved": resolved_count}


@router.post("/jobs/daily")
async def trigger_daily_job():
    """
    Trigger the daily job on-demand.

    Used by external schedulers (e.g., Render Cron Jobs) to run the daily
    pick generation and outcome resolution.
    """
    await run_daily_job()
    return {"message": "Daily job completed successfully"}


@router.post("/simulation/regenerate")
async def regenerate_picks(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD), defaults to start_date"),
    db: AsyncSession = Depends(get_db),
):
    """
    Regenerate picks for a date range, then resolve outcomes.

    This endpoint:
    1. Deletes existing picks for each date
    2. Regenerates picks using ML models and current odds (historical odds may not be available)
    3. Resolves outcomes from ESPN game results

    Use this to backfill historical dates where picks were deleted.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else start

    picks_repo = PicksRepository(db)
    picks_service = PicksService(db)
    results = []

    current = start
    while current <= end:
        # Step 1: Delete existing picks for this date
        deleted = await picks_repo.delete_all_for_date(current)

        # Step 2: Regenerate picks using ML models
        picks = await picks_service.generate_picks_for_date(current)

        # Step 3: Resolve outcomes from ESPN
        await resolve_all_picks(db, current)

        results.append({
            "date": str(current),
            "deleted": deleted,
            "generated": len(picks),
        })
        current += timedelta(days=1)

    return {"message": f"Regenerated picks for {len(results)} date(s)", "results": results}
