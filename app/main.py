import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes import picks, simulation
from app.jobs.daily_job import scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if settings.ENVIRONMENT == "production":
        scheduler.start()
        # Startup catch-up: resolve missed picks and refresh stale snapshots.
        # This handles the case where Render slept through scheduled cron jobs.
        asyncio.create_task(_startup_catchup())
    yield
    # Shutdown
    if settings.ENVIRONMENT == "production":
        scheduler.shutdown()


async def _startup_catchup():
    """
    Background task that runs after server startup to catch up on
    any missed work (unresolved picks, stale snapshots).

    Non-blocking: the server accepts requests immediately while
    this runs in the background.
    """
    # Let the server fully initialize (DB pool, routes, etc.)
    await asyncio.sleep(5)

    logger.info("=== STARTUP CATCH-UP BEGINNING ===")

    try:
        from datetime import datetime, timedelta
        from sqlalchemy import select
        import pytz

        from app.jobs.daily_job import resolve_all_pending_picks, run_hourly_picks_job
        from app.db.database import async_session_factory
        from app.db.models import PicksSnapshot

        EST = pytz.timezone("America/New_York")

        # Step 1: Resolve all pending picks from past dates
        resolved_count = await resolve_all_pending_picks()
        logger.info(f"Startup catch-up: resolved {resolved_count} pending picks")

        # Step 2: Check if picks snapshot predates the last SCHEDULED run time
        # Picks only update at 9 AM and 6 PM EST - only catch up if one was missed
        # This prevents spurious refreshes when Render restarts mid-day
        async with async_session_factory() as session:
            result = await session.execute(
                select(PicksSnapshot)
                .order_by(PicksSnapshot.computed_at.desc())
                .limit(1)
            )
            latest_snapshot = result.scalar_one_or_none()

            now = datetime.now(EST)
            today_9am = now.replace(hour=9, minute=0, second=0, microsecond=0)
            today_6pm = now.replace(hour=18, minute=0, second=0, microsecond=0)

            # Find the most recent scheduled time that has already passed today
            if now >= today_6pm:
                last_scheduled = today_6pm
            elif now >= today_9am:
                last_scheduled = today_9am
            else:
                last_scheduled = None  # Before 9 AM, nothing to catch up

            snapshot_stale = True  # default

            if last_scheduled is None:
                # Before 9 AM, no scheduled run has occurred yet today
                snapshot_stale = False
                logger.info("Before 9 AM - no catch-up needed")
            elif latest_snapshot and latest_snapshot.computed_at:
                snapshot_time = latest_snapshot.computed_at.astimezone(EST)
                snapshot_stale = snapshot_time < last_scheduled
                logger.info(f"Last scheduled: {last_scheduled.strftime('%I:%M %p')}, "
                            f"snapshot: {snapshot_time.strftime('%I:%M %p')}, stale: {snapshot_stale}")
            else:
                logger.info("No picks snapshot found - will regenerate")
                snapshot_stale = True

        # Step 3: Refresh picks snapshot only if a scheduled run was missed
        if snapshot_stale:
            logger.info("Refreshing picks snapshot (scheduled run was missed)...")
            await run_hourly_picks_job()
            logger.info("Picks snapshot refreshed")
        else:
            logger.info("Picks snapshot is current - no refresh needed")

        logger.info("=== STARTUP CATCH-UP COMPLETE ===")

    except Exception as e:
        logger.error(f"Startup catch-up failed: {e}", exc_info=True)
        # Don't crash the server - catch-up is best-effort


app = FastAPI(
    title="EdgeBet API",
    description="NBA betting analytics platform with ML-powered predictions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(picks.router, tags=["picks"])
app.include_router(simulation.router, tags=["simulation"])


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}
