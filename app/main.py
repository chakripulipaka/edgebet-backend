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

        # Step 2: Check if picks snapshot is stale (> 1 hour old)
        async with async_session_factory() as session:
            result = await session.execute(
                select(PicksSnapshot)
                .order_by(PicksSnapshot.computed_at.desc())
                .limit(1)
            )
            latest_snapshot = result.scalar_one_or_none()

            now = datetime.now(EST)
            snapshot_stale = True

            if latest_snapshot and latest_snapshot.computed_at:
                snapshot_age = now - latest_snapshot.computed_at.astimezone(EST)
                snapshot_stale = snapshot_age > timedelta(hours=1)
                logger.info(f"Latest picks snapshot age: {snapshot_age}, stale: {snapshot_stale}")
            else:
                logger.info("No picks snapshot found - will regenerate")

        # Step 3: Refresh hourly picks snapshot if stale
        if snapshot_stale:
            logger.info("Refreshing picks snapshot (stale or missing)...")
            await run_hourly_picks_job()
            logger.info("Picks snapshot refreshed")

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
