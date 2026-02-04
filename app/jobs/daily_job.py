"""Daily job for updating picks and simulation state."""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any
import pytz

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db.database import async_session_factory
from app.db.repositories.games import GameRepository
from app.db.repositories.picks import PicksRepository
from app.services.picks_service import PicksService
from app.services.simulation_service import SimulationService
from app.services.espn_data import espn_data_service
from app.services.odds_data import odds_service, clear_odds_cache
from app.config import settings

logger = logging.getLogger(__name__)

# Timezone for scheduling (EST)
EST = pytz.timezone("America/New_York")

# Create scheduler
scheduler = AsyncIOScheduler()


async def resolve_all_pending_picks() -> int:
    """
    Resolve ALL unresolved picks across all past dates.

    This is the safety net that ensures no picks are ever lost,
    regardless of how many days the server was asleep or how many
    cron jobs were missed.

    Returns the total number of picks resolved.
    """
    logger.info("Starting resolve_all_pending_picks...")

    async with async_session_factory() as session:
        try:
            picks_repo = PicksRepository(session)
            pending = await picks_repo.get_pending_picks()

            if not pending:
                logger.info("No pending picks to resolve")
                return 0

            today = date.today()
            past_dates = sorted(set(
                p.pick_date for p in pending
                if p.pick_date < today  # Skip today (games may still be live)
            ))

            if not past_dates:
                logger.info("All pending picks are for today or future - skipping")
                return 0

            logger.info(
                f"Found {len(pending)} unresolved picks across "
                f"{len(past_dates)} past date(s): {past_dates}"
            )

            total_resolved = 0
            for pick_date in past_dates:
                before = await picks_repo.get_pending_picks_for_date(pick_date)
                await resolve_all_picks(session, pick_date)
                after = await picks_repo.get_pending_picks_for_date(pick_date)
                resolved = len(before) - len(after)
                total_resolved += resolved

                if after:
                    logger.warning(
                        f"{len(after)} picks still pending for {pick_date} "
                        f"(games may not be final on ESPN yet)"
                    )

            # Recompute simulation snapshot if any picks were resolved
            if total_resolved > 0:
                yesterday = today - timedelta(days=1)
                logger.info(f"Recomputing simulation snapshot ({total_resolved} picks resolved)...")
                await compute_and_store_snapshot(session, yesterday)

            logger.info(f"resolve_all_pending_picks complete: resolved {total_resolved} picks")
            return total_resolved

        except Exception as e:
            logger.error(f"resolve_all_pending_picks failed: {e}", exc_info=True)
            raise


async def run_daily_job():
    """
    Main daily job that runs at 9 AM EST.

    Steps:
    1. Resolve ALL pending picks (not just yesterday - catches up on missed days)
    2. Generate today's picks
    3. Compute and store simulation snapshot
    """
    logger.info("Starting daily job...")

    async with async_session_factory() as session:
        try:
            today = date.today()
            yesterday = today - timedelta(days=1)

            picks_repo = PicksRepository(session)

            # Step 1: Resolve ALL pending picks from past dates
            # This catches up on any missed days, not just yesterday
            pending = await picks_repo.get_pending_picks()
            past_dates = sorted(set(
                p.pick_date for p in pending
                if p.pick_date < today
            ))

            if past_dates:
                logger.info(f"Resolving pending picks for {len(past_dates)} date(s): {past_dates}")
                for pick_date in past_dates:
                    await resolve_all_picks(session, pick_date)
            else:
                logger.info("No pending picks to resolve")

            # Step 2: Generate today's picks (if not already generated)
            existing_picks = await picks_repo.get_all_by_date(today)
            if not existing_picks:
                logger.info(f"Generating picks for {today}")
                picks_service = PicksService(session)
                await picks_service.generate_picks_for_date(today)

            # Step 3: Compute and store simulation snapshot
            logger.info("Computing simulation snapshot...")
            await compute_and_store_snapshot(session, yesterday)

            logger.info("Daily job completed successfully")

        except Exception as e:
            logger.error(f"Daily job failed: {e}", exc_info=True)
            raise


async def run_hourly_picks_job():
    """
    Hourly job to pre-compute picks for instant dashboard loading.
    Runs at :00 of each hour EST.

    - Fetches fresh odds for scheduled games only
    - Updates picks for games that haven't started
    - Creates snapshot for instant frontend loading

    Games that have started keep their locked odds (no update).
    """
    logger.info("Starting hourly picks job...")

    async with async_session_factory() as session:
        try:
            from app.services.picks_service import get_team_name
            from app.db.models import PicksSnapshot

            now = datetime.now(EST)
            picks_service = PicksService(session)
            picks_repo = PicksRepository(session)

            # Clear caches to get fresh data
            clear_odds_cache()

            # Clear the processed dates cache so we regenerate picks
            from app.services.picks_service import _processed_dates_cache
            _processed_dates_cache.clear()
            logger.info("Cleared odds and processed dates caches")

            # Collect all picks and track which games have picks (across all dates)
            all_picks: List[Dict[str, Any]] = []
            all_games_with_picks: set = set()  # Track ESPN game IDs that have picks
            all_espn_games: List[Dict[str, Any]] = []  # All games from ESPN

            # Process each date in the window (today, tomorrow, day after)
            for days_ahead in [0, 1, 2]:
                target_date = (now + timedelta(days=days_ahead)).date()
                logger.info(f"Processing {target_date} (days ahead: {days_ahead})")

                # Get existing picks for this date
                existing_picks = await picks_repo.get_all_by_date(target_date)

                # Separate live/final picks from scheduled picks
                live_final_picks = []
                scheduled_pick_ids = []

                for pick in existing_picks:
                    status = picks_service._calculate_game_status(pick.game_time)
                    if status == "scheduled":
                        # Will regenerate with fresh odds
                        scheduled_pick_ids.append(pick.id)
                    else:
                        # Keep locked odds for live/final games
                        formatted = picks_service._format_pick(pick, status)
                        live_final_picks.append(formatted)
                        all_games_with_picks.add(pick.espn_game_id)

                # Delete scheduled picks (they'll be regenerated with fresh odds)
                if scheduled_pick_ids:
                    await picks_repo.delete_by_ids(scheduled_pick_ids)
                    logger.info(f"Deleted {len(scheduled_pick_ids)} scheduled picks for {target_date}")

                # Add live/final picks to our collection
                all_picks.extend(live_final_picks)

                # Generate fresh picks for scheduled games (fetches fresh odds)
                new_picks = await picks_service.generate_picks_for_date(target_date)

                # Add only scheduled picks (live/final already added above)
                for pick in new_picks:
                    if pick.get("gameStatus") == "scheduled":
                        all_picks.append(pick)
                        if pick.get("espnGameId"):
                            all_games_with_picks.add(pick.get("espnGameId"))

                # Get all games from ESPN for this date
                games_on_date = espn_data_service.get_games_for_date(target_date)
                for game in games_on_date:
                    all_espn_games.append({
                        "game_id": game.get("nba_game_id"),
                        "home_team_id": game.get("home_team_id"),
                        "away_team_id": game.get("away_team_id"),
                        "game_time": game.get("game_time"),
                        "date": target_date,
                    })

            # Find games without picks (scheduled games with no odds available)
            games_without_picks = []
            for game in all_espn_games:
                game_id = game.get("game_id")
                game_time = game.get("game_time")

                # Skip if we have picks for this game
                if game_id in all_games_with_picks:
                    continue

                # Calculate game status
                if game_time:
                    status = picks_service._calculate_game_status(game_time)
                else:
                    status = "scheduled"

                # Only include scheduled games (live/final without picks are edge cases)
                if status == "scheduled":
                    games_without_picks.append({
                        "homeTeam": get_team_name(game.get("home_team_id")),
                        "awayTeam": get_team_name(game.get("away_team_id")),
                        "gameTime": game_time.isoformat() if game_time else f"{game.get('date')}T19:00:00",
                        "gameStatus": "scheduled",
                        "espnGameId": game_id,
                    })

            # Filter to 36-hour window
            cutoff_time = now + timedelta(hours=36)

            def parse_time(time_str):
                try:
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    if dt.tzinfo:
                        return dt.astimezone(EST)
                    return dt.replace(tzinfo=EST.localize(datetime.now()).tzinfo)
                except Exception:
                    return now + timedelta(days=365)

            # Filter picks to 36-hour window (keep live games always)
            filtered_picks = []
            for pick in all_picks:
                game_time = parse_time(pick.get("gameTime", ""))
                game_status = pick.get("gameStatus", "scheduled")

                if game_status in ("in_progress", "halftime"):
                    # Always show live games
                    filtered_picks.append(pick)
                elif game_status == "scheduled" and game_time <= cutoff_time:
                    # Scheduled games within window
                    filtered_picks.append(pick)
                # Skip final games (they disappear from dashboard)

            # Filter games without picks to 36-hour window
            filtered_games_without = [
                g for g in games_without_picks
                if parse_time(g.get("gameTime", "")) <= cutoff_time
            ]

            # Sort picks: live first, then by game time
            def sort_key(pick):
                status = pick.get("gameStatus", "scheduled")
                is_live = status in ("in_progress", "halftime")
                game_time = parse_time(pick.get("gameTime", ""))
                return (0 if is_live else 1, game_time)

            filtered_picks.sort(key=sort_key)
            filtered_games_without.sort(key=lambda g: parse_time(g.get("gameTime", "")))

            # Check if all games are complete (nothing to show)
            all_complete = len(filtered_picks) == 0 and len(filtered_games_without) == 0

            # Create snapshot
            snapshot_data = {
                "picks": filtered_picks,
                "gamesWithoutPicks": filtered_games_without,
                "allGamesComplete": all_complete,
                "computedAt": now.isoformat(),
            }

            snapshot = PicksSnapshot(
                computed_at=now,
                snapshot_data=snapshot_data,
            )

            session.add(snapshot)
            await session.commit()

            logger.info(
                f"Hourly picks job complete: {len(filtered_picks)} picks, "
                f"{len(filtered_games_without)} games without picks"
            )

        except Exception as e:
            logger.error(f"Hourly picks job failed: {e}", exc_info=True)
            raise


async def compute_and_store_snapshot(session, through_date: date):
    """
    Compute full simulation and store as a snapshot for fast retrieval.

    Args:
        session: Database session
        through_date: The last date included in the simulation data
    """
    from app.db.models import SimulationSnapshot

    sim_service = SimulationService(session)

    # Run the full simulation calculation
    result = await sim_service.run_live_simulation(starting_bankroll=100.0)

    # Create snapshot with the full response data
    snapshot = SimulationSnapshot(
        computed_at=datetime.now(EST),
        through_date=through_date,
        snapshot_data=result
    )

    session.add(snapshot)
    await session.commit()

    logger.info(
        f"Stored simulation snapshot: through_date={through_date}, "
        f"final_bankroll=${result.get('finalBankroll', 0):.2f}, "
        f"days_simulated={result.get('daysSimulated', 0)}"
    )


async def resolve_all_picks(session, pick_date: date):
    """
    Resolve ALL picks for a date and store their outcomes.

    Steps:
    1. Get all picks for the date
    2. For each pick, fetch game result from ESPN
    3. Store home_score, away_score, and outcome for each pick
    4. Keep ALL picks (no deletion) for simulation with all bets
    """
    from app.services.simulation_service import determine_outcome

    picks_repo = PicksRepository(session)
    picks = await picks_repo.get_all_by_date(pick_date)

    if not picks:
        logger.info(f"No picks found for {pick_date}")
        return

    logger.info(f"Resolving {len(picks)} picks for {pick_date}")

    # Cache game results to avoid duplicate ESPN calls for same game
    game_results_cache = {}
    resolved_count = 0

    for pick in picks:
        # Skip if already resolved
        if pick.outcome is not None:
            logger.debug(f"Pick {pick.id} already resolved: {pick.outcome}")
            continue

        # Check cache first, then fetch from ESPN
        if pick.espn_game_id not in game_results_cache:
            game_result = espn_data_service.get_game_result(pick.espn_game_id)
            game_results_cache[pick.espn_game_id] = game_result
        else:
            game_result = game_results_cache[pick.espn_game_id]

        if not game_result or game_result.get("status") != "final":
            logger.warning(f"Game not final yet for pick {pick.id} (game {pick.espn_game_id})")
            continue

        home_score = game_result.get("home_score")
        away_score = game_result.get("away_score")

        if home_score is None or away_score is None:
            logger.warning(f"Scores not available for pick {pick.id}")
            continue

        # Determine outcome
        outcome, description = determine_outcome(pick, home_score, away_score)

        # Store outcome with scores
        await picks_repo.update_outcome_with_scores(pick, home_score, away_score, outcome)
        resolved_count += 1
        logger.info(f"Resolved pick {pick.id}: {pick.side} ({pick.bet_type}) -> {outcome}")

    logger.info(f"Resolved {resolved_count} picks for {pick_date}")


# Keep old function name as alias for backwards compatibility with backfill endpoint
async def resolve_and_cleanup_picks(session, pick_date: date):
    """Alias for resolve_all_picks (backwards compatibility)."""
    await resolve_all_picks(session, pick_date)


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

    # Hourly picks refresh at :00 of each hour
    # Pre-computes picks for instant dashboard loading
    scheduler.add_job(
        run_hourly_picks_job,
        CronTrigger(minute=0, timezone=EST),  # Every hour at :00
        id="hourly_picks_job",
        replace_existing=True,
    )
