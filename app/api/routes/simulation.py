"""API routes for simulation endpoints."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import SimulationSnapshot
from app.db.repositories.picks import PicksRepository
from app.api.schemas.simulation import SimulationResponse
from app.services.picks_service import PicksService
from app.jobs.daily_job import resolve_and_cleanup_picks, resolve_all_picks, run_daily_job, run_hourly_picks_job

router = APIRouter()


@router.get("/simulation", response_model=SimulationResponse)
async def get_simulation(
    starting_bankroll: float = Query(100.0, description="Starting portfolio value"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get pre-computed simulation data from the latest snapshot.

    Snapshots are computed daily at 9 AM EST.
    If no snapshot exists yet, falls back to live calculation.

    Returns bankroll progression, win/loss record, and bet history.
    """
    # Try to get the latest snapshot (fast path)
    result = await db.execute(
        select(SimulationSnapshot)
        .order_by(SimulationSnapshot.computed_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if snapshot:
        # Return pre-computed data (instant)
        return SimulationResponse(**snapshot.snapshot_data)

    # No snapshot yet - return empty simulation data
    return SimulationResponse(
        chartData=[],
        dailySummaries=[],
        finalBankroll=starting_bankroll,
        totalWins=0,
        totalLosses=0,
        totalPushes=0,
        winRate=0.0,
        roi=0.0,
        maxDrawdown=0.0,
        peakBankroll=starting_bankroll,
        daysSimulated=0,
        totalBets=0,
        startingBankroll=starting_bankroll,
    )


@router.post("/simulation/resolve-pending")
async def resolve_pending_picks(db: AsyncSession = Depends(get_db)):
    """
    Resolve outcomes for picks where games have completed.

    Called by frontend auto-polling to check for newly completed games.
    Only resolves picks where games are now final but outcome is not yet stored.
    """
    picks_repo = PicksRepository(db)
    pending_picks = await picks_repo.get_pending_picks()

    if not pending_picks:
        return {"message": "No pending picks", "resolved": 0, "pending": 0}

    # Get unique dates (only check recent dates to avoid unnecessary API calls)
    dates = sorted(set(p.pick_date for p in pending_picks))
    resolved_count = 0
    still_pending = 0

    for pick_date in dates:
        # Only process dates within the last 7 days
        if (datetime.now().date() - pick_date).days > 7:
            continue

        await resolve_all_picks(db, pick_date)

        # Check how many were resolved vs still pending
        remaining = await picks_repo.get_pending_picks_for_date(pick_date)
        if not remaining:
            resolved_count += 1
        else:
            still_pending += len(remaining)

    return {
        "message": f"Processed {len(dates)} date(s)",
        "dates_resolved": resolved_count,
        "picks_still_pending": still_pending,
    }


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


@router.delete("/simulation/clear-all")
async def clear_all_picks(db: AsyncSession = Depends(get_db)):
    """
    Delete ALL picks and simulation data to start fresh.

    Use this when you need to reset and start tracking from scratch
    (e.g., after fixing data quality issues).
    """
    from sqlalchemy import delete
    from app.db.models import DailyPick, SimulationState

    # Delete all picks
    result = await db.execute(delete(DailyPick))
    picks_deleted = result.rowcount

    # Delete all simulation states
    result = await db.execute(delete(SimulationState))
    states_deleted = result.rowcount

    await db.commit()

    return {
        "message": "All data cleared",
        "picks_deleted": picks_deleted,
        "simulation_states_deleted": states_deleted,
    }


@router.post("/jobs/daily")
async def trigger_daily_job():
    """
    Trigger the daily job on-demand.

    Used by external schedulers (e.g., Render Cron Jobs) to run the daily
    pick generation and outcome resolution.
    """
    await run_daily_job()
    return {"message": "Daily job completed successfully"}


@router.post("/jobs/hourly-picks")
async def trigger_hourly_picks_job():
    """
    Trigger the hourly picks job on-demand.

    Used by external schedulers (e.g., Render Cron Jobs) to pre-compute
    picks snapshot for instant dashboard loading.

    This job:
    - Fetches fresh odds for scheduled games
    - Keeps locked odds for live/final games
    - Creates a snapshot for instant frontend retrieval
    """
    await run_hourly_picks_job()
    return {"message": "Hourly picks job completed successfully"}


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
