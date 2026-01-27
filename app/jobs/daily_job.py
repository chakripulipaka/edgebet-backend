"""Daily job for updating picks and simulation state."""

import logging
from datetime import date, datetime, timedelta
import pytz

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db.database import async_session_factory
from app.db.repositories.games import GameRepository
from app.db.repositories.picks import PicksRepository
from app.services.picks_service import PicksService
from app.services.simulation_service import SimulationService
from app.services.espn_data import espn_data_service
from app.config import settings

logger = logging.getLogger(__name__)

# Timezone for scheduling (EST)
EST = pytz.timezone("America/New_York")

# Create scheduler
scheduler = AsyncIOScheduler()


async def run_daily_job():
    """
    Main daily job that runs at 9 AM EST.

    Steps:
    1. Update yesterday's game outcomes
    2. Resolve yesterday's top pick and store outcome with scores
    3. Clean up non-top picks for yesterday
    4. Generate today's picks
    """
    logger.info("Starting daily job...")

    async with async_session_factory() as session:
        try:
            today = date.today()
            yesterday = today - timedelta(days=1)

            picks_repo = PicksRepository(session)

            # Step 1 & 2: Resolve yesterday's top pick and store outcome
            logger.info(f"Processing picks for {yesterday}")
            await resolve_and_cleanup_picks(session, yesterday)

            # Step 3: Generate today's picks (if not already generated)
            existing_picks = await picks_repo.get_all_by_date(today)
            if not existing_picks:
                logger.info(f"Generating picks for {today}")
                picks_service = PicksService(session)
                await picks_service.generate_picks_for_date(today)

            logger.info("Daily job completed successfully")

        except Exception as e:
            logger.error(f"Daily job failed: {e}", exc_info=True)
            raise


async def resolve_and_cleanup_picks(session, pick_date: date):
    """
    Resolve the top-edge pick for a date and clean up other picks.

    Steps:
    1. Get all picks for the date
    2. Find the top-edge pick
    3. Fetch game result from ESPN
    4. Store home_score, away_score, and outcome
    5. Delete all other picks for this date
    """
    from app.services.simulation_service import determine_outcome

    picks_repo = PicksRepository(session)
    picks = await picks_repo.get_all_by_date(pick_date)

    if not picks:
        logger.info(f"No picks found for {pick_date}")
        return

    # Find top-edge pick
    top_pick = max(picks, key=lambda p: float(p.edge))
    logger.info(f"Top pick for {pick_date}: {top_pick.side} ({top_pick.bet_type}) with edge {float(top_pick.edge):.1f}%")

    # Check if already resolved
    if top_pick.outcome is not None:
        logger.info(f"Pick already resolved: {top_pick.outcome}")
        # Still clean up other picks
        deleted = await picks_repo.delete_picks_for_date_except(pick_date, top_pick.id)
        logger.info(f"Cleaned up {deleted} non-top picks")
        return

    # Fetch game result from ESPN
    game_result = espn_data_service.get_game_result(top_pick.espn_game_id)

    if not game_result or game_result.get("status") != "final":
        logger.warning(f"Game not final yet for pick {top_pick.id}")
        return

    home_score = game_result.get("home_score")
    away_score = game_result.get("away_score")

    if home_score is None or away_score is None:
        logger.warning(f"Scores not available for pick {top_pick.id}")
        return

    # Determine outcome
    outcome, description = determine_outcome(top_pick, home_score, away_score)
    logger.info(f"Outcome: {outcome} - {description}")

    # Store outcome with scores
    await picks_repo.update_outcome_with_scores(top_pick, home_score, away_score, outcome)
    logger.info(f"Stored outcome for pick {top_pick.id}")

    # Clean up other picks
    deleted = await picks_repo.delete_picks_for_date_except(pick_date, top_pick.id)
    logger.info(f"Cleaned up {deleted} non-top picks for {pick_date}")


async def update_game_outcomes(session, game_date: date):
    """
    Update game scores from ESPN API for completed games.

    Args:
        session: Database session
        game_date: Date to update
    """
    games_repo = GameRepository(session)
    games = await games_repo.get_by_date(game_date)

    if not games:
        return

    # Fetch fresh data from ESPN API
    api_games = espn_data_service.get_games_for_date(game_date)

    for game in games:
        if game.status == "final":
            continue

        # Find matching API game
        for api_game in api_games:
            if api_game["nba_game_id"] == game.nba_game_id:
                if api_game.get("home_score") is not None:
                    await games_repo.update_score(
                        game,
                        api_game["home_score"],
                        api_game["away_score"],
                        api_game.get("status", "final"),
                    )
                    logger.info(f"Updated game {game.nba_game_id}: {api_game['home_score']}-{api_game['away_score']}")
                break


async def sync_today_games():
    """
    Sync today's scheduled games from ESPN API.
    """
    logger.info("Syncing today's games...")

    async with async_session_factory() as session:
        try:
            from app.db.repositories.teams import TeamRepository

            games_repo = GameRepository(session)
            teams_repo = TeamRepository(session)
            today = date.today()

            # Get games from ESPN API
            api_games = espn_data_service.get_games_for_date(today)

            for api_game in api_games:
                # Check if game already exists
                existing = await games_repo.get_by_nba_id(api_game["nba_game_id"])
                if existing:
                    continue

                # Get team IDs
                home_team = await teams_repo.get_by_nba_id(api_game["home_team_id"])
                away_team = await teams_repo.get_by_nba_id(api_game["away_team_id"])

                if not home_team or not away_team:
                    logger.warning(f"Teams not found for game {api_game['nba_game_id']}")
                    continue

                # Create game
                await games_repo.create(
                    nba_game_id=api_game["nba_game_id"],
                    season="2024-25",
                    game_date=api_game["game_date"],
                    game_time=api_game.get("game_time"),
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    status=api_game.get("status", "scheduled"),
                )
                logger.info(f"Created game: {home_team.abbreviation} vs {away_team.abbreviation}")

        except Exception as e:
            logger.error(f"Failed to sync games: {e}", exc_info=True)


# Schedule jobs
if settings.ENVIRONMENT == "production":
    # Main daily job at 9 AM EST
    scheduler.add_job(
        run_daily_job,
        CronTrigger(hour=9, minute=0, timezone=EST),
        id="daily_job",
        replace_existing=True,
    )

    # Sync games at 6 AM EST
    scheduler.add_job(
        sync_today_games,
        CronTrigger(hour=6, minute=0, timezone=EST),
        id="sync_games",
        replace_existing=True,
    )
