"""CLI commands for EdgeBet backend management."""

import asyncio
import argparse
import logging
from datetime import date, datetime, timedelta
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def init_db():
    """Initialize database and run migrations."""
    from alembic.config import Config
    from alembic import command

    logger.info("Initializing database...")

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    logger.info("Database initialized successfully")


async def sync_teams():
    """Sync all NBA teams to database."""
    from app.db.database import async_session_factory
    from app.db.repositories.teams import TeamRepository
    from app.services.espn_data import espn_data_service

    logger.info("Syncing NBA teams...")

    async with async_session_factory() as session:
        teams_repo = TeamRepository(session)
        teams = espn_data_service.get_all_teams()

        for team in teams:
            await teams_repo.upsert(
                nba_team_id=team["nba_team_id"],
                abbreviation=team["abbreviation"],
                name=team["name"],
                city=team["city"],
            )
            logger.info(f"Upserted team: {team['abbreviation']}")

    logger.info(f"Synced {len(teams)} teams")


async def sync_historical_data(start_season: str = "2022-23", end_season: str = "2024-25"):
    """
    Sync historical games and stats from NBA API.

    Args:
        start_season: Starting season (e.g., "2022-23")
        end_season: Ending season (e.g., "2024-25")
    """
    from app.db.database import async_session_factory
    from app.db.repositories.teams import TeamRepository
    from app.db.repositories.games import GameRepository
    from app.db.repositories.stats import StatsRepository
    from app.services.espn_data import espn_data_service

    logger.info(f"Syncing historical data from {start_season} to {end_season}...")

    # Parse seasons
    seasons = []
    start_year = int(start_season.split("-")[0])
    end_year = int(end_season.split("-")[0])

    for year in range(start_year, end_year + 1):
        seasons.append(f"{year}-{str(year + 1)[-2:]}")

    async with async_session_factory() as session:
        teams_repo = TeamRepository(session)
        games_repo = GameRepository(session)
        stats_repo = StatsRepository(session)

        # First sync teams
        await sync_teams()

        total_games = 0

        for season in seasons:
            logger.info(f"Processing season {season}...")

            # Get games for the season
            games = espn_data_service.get_games_for_season(season)
            logger.info(f"Found {len(games)} games for {season}")

            for game_data in games:
                # Skip if game already exists
                existing = await games_repo.get_by_nba_id(game_data["nba_game_id"])
                if existing:
                    continue

                # Get team database IDs
                home_team = await teams_repo.get_by_nba_id(game_data["home_team_id"])
                away_team = await teams_repo.get_by_nba_id(game_data["away_team_id"])

                if not home_team or not away_team:
                    logger.warning(f"Teams not found for game {game_data['nba_game_id']}")
                    continue

                # Create game
                game = await games_repo.create(
                    nba_game_id=game_data["nba_game_id"],
                    season=game_data["season"],
                    game_date=game_data["game_date"],
                    game_time=game_data.get("game_time"),
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    status=game_data.get("status", "final"),
                )

                # Update scores if available
                if game_data.get("home_score") is not None:
                    await games_repo.update_score(
                        game,
                        game_data["home_score"],
                        game_data["away_score"],
                        "final",
                    )

                    # Create basic stats
                    await stats_repo.upsert(
                        game_id=game.id,
                        team_id=home_team.id,
                        is_home=True,
                        points=game_data["home_score"],
                        opponent_points=game_data["away_score"],
                    )
                    await stats_repo.upsert(
                        game_id=game.id,
                        team_id=away_team.id,
                        is_home=False,
                        points=game_data["away_score"],
                        opponent_points=game_data["home_score"],
                    )

                total_games += 1

                if total_games % 100 == 0:
                    logger.info(f"Processed {total_games} games...")

    logger.info(f"Synced {total_games} total games")


async def sync_week_stats(days: int = 7):
    """
    Sync last N days of games with full box score stats.

    Args:
        days: Number of days to sync (default: 7)
    """
    from app.db.database import async_session_factory
    from app.db.repositories.teams import TeamRepository
    from app.db.repositories.games import GameRepository
    from app.db.repositories.stats import StatsRepository
    from app.services.espn_data import espn_data_service

    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days - 1)

    logger.info(f"Syncing games from {start_date} to {end_date}")

    async with async_session_factory() as session:
        teams_repo = TeamRepository(session)
        games_repo = GameRepository(session)
        stats_repo = StatsRepository(session)

        total_games = 0
        current_date = start_date

        while current_date <= end_date:
            logger.info(f"Fetching games for {current_date}...")
            games = espn_data_service.get_games_for_date(current_date)
            logger.info(f"Found {len(games)} games for {current_date}")

            for game_data in games:
                # Get DB team IDs
                home_team = await teams_repo.get_by_nba_id(game_data["home_team_id"])
                away_team = await teams_repo.get_by_nba_id(game_data["away_team_id"])

                if not home_team or not away_team:
                    logger.warning(
                        f"Teams not found for game {game_data['nba_game_id']} "
                        f"(home: {game_data['home_team_id']}, away: {game_data['away_team_id']})"
                    )
                    continue

                # Get or create game
                game = await games_repo.get_by_nba_id(game_data["nba_game_id"])
                if not game:
                    game = await games_repo.create(
                        nba_game_id=game_data["nba_game_id"],
                        season="2024-25",
                        game_date=current_date,
                        game_time=game_data.get("game_time"),
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        status=game_data.get("status", "final"),
                    )

                # Try to fetch box score for all games - let the API tell us if unavailable
                logger.info(f"Fetching box score for game {game_data['nba_game_id']}...")
                box_score = espn_data_service.get_box_score(game_data["nba_game_id"])
                logger.info(f"Box score result: {box_score}")

                # If box score is empty, try fallback method using LeagueGameFinder
                if not box_score:
                    logger.info(f"Box score empty, trying fallback for game {game_data['nba_game_id']}...")
                    box_score = espn_data_service.get_game_stats_fallback(game_data["nba_game_id"])
                    logger.info(f"Fallback result: {box_score}")

                if not box_score:
                    logger.info(f"No stats available for game {game_data['nba_game_id']} from any source")
                    continue

                # Get scores from box score (more reliable than scoreboard API)
                home_team_stats = box_score.get(game_data["home_team_id"], {})
                away_team_stats = box_score.get(game_data["away_team_id"], {})
                logger.info(f"Home stats: {home_team_stats}, Away stats: {away_team_stats}")

                if home_team_stats.get("points") and away_team_stats.get("points"):
                    await games_repo.update_score(
                        game,
                        home_team_stats["points"],
                        away_team_stats["points"],
                        "final"
                    )

                # Store stats for both teams
                for nba_team_id, stats in box_score.items():
                    db_team = await teams_repo.get_by_nba_id(nba_team_id)
                    if not db_team:
                        logger.warning(f"Team not found in DB: {nba_team_id}")
                        continue

                    is_home = nba_team_id == game_data["home_team_id"]
                    opponent_nba_id = (
                        game_data["away_team_id"] if is_home else game_data["home_team_id"]
                    )
                    opponent_stats = box_score.get(opponent_nba_id, {})

                    # Default rest days - can be calculated more precisely
                    # once we have more historical data loaded
                    rest_days = 3

                    # Use existing upsert method with all fields
                    logger.info(f"Upserting stats for team {db_team.abbreviation} (game_id={game.id}, team_id={db_team.id})")
                    await stats_repo.upsert(
                        game_id=game.id,
                        team_id=db_team.id,
                        is_home=is_home,
                        points=stats.get("points"),
                        opponent_points=opponent_stats.get("points"),
                        fg_pct=stats.get("fg_pct"),
                        fg3_pct=stats.get("fg3_pct"),
                        ft_pct=stats.get("ft_pct"),
                        assists=stats.get("assists"),
                        rebounds=stats.get("rebounds"),
                        blocks=stats.get("blocks"),
                        steals=stats.get("steals"),
                        turnovers=stats.get("turnovers"),
                        pace=stats.get("pace"),
                        rest_days=rest_days,
                    )
                    logger.info(f"Successfully upserted stats for team {db_team.abbreviation}")

                total_games += 1
                if total_games % 10 == 0:
                    logger.info(f"Processed {total_games} games...")

            current_date += timedelta(days=1)

    logger.info(f"Sync complete. Processed {total_games} games.")


def train_models(start_season: int = 2020, end_season: int = 2024):
    """
    Train all ML models using Kaggle SQLite data directly.

    Args:
        start_season: Starting season year (e.g., 2020 for 2020-21)
        end_season: Ending season year (e.g., 2024 for 2024-25)
    """
    from cli.kaggle_trainer import train_from_kaggle

    logger.info(f"Training models from Kaggle data (seasons {start_season}-{end_season})...")

    results = train_from_kaggle(start_season, end_season)

    if "error" in results:
        logger.error(f"Training failed: {results['error']}")
        return

    logger.info("Training results:")
    for model_name, metrics in results.items():
        logger.info(f"  {model_name}: {metrics}")

    logger.info("Models trained and saved successfully")


async def run_daily_job_manual():
    """Manually run the daily job."""
    from app.jobs.daily_job import run_daily_job

    logger.info("Running daily job manually...")
    await run_daily_job()
    logger.info("Daily job completed")


async def generate_mock_data():
    """Generate mock simulation data for testing."""
    from decimal import Decimal
    from app.db.database import async_session_factory
    from app.db.repositories.simulation import SimulationRepository
    from app.db.repositories.picks import PicksRepository
    from app.db.repositories.games import GameRepository
    from app.core.constants import INITIAL_BANKROLL
    import random

    logger.info("Generating mock simulation data...")

    async with async_session_factory() as session:
        sim_repo = SimulationRepository(session)

        today = date.today()
        bankroll = INITIAL_BANKROLL
        peak = bankroll
        max_dd = 0.0
        wins = 0
        losses = 0

        # Generate 30 days of simulation data
        for i in range(30):
            state_date = today - timedelta(days=30 - i)

            # Simulate a bet result
            won = random.random() < 0.55  # 55% win rate
            kelly = random.uniform(0.03, 0.08)  # 3-8% bet size
            odds_decimal = 1.9 if won else 0

            if won:
                wins += 1
                bankroll += bankroll * kelly * (odds_decimal - 1)
            else:
                losses += 1
                bankroll -= bankroll * kelly

            peak = max(peak, bankroll)
            drawdown = ((peak - bankroll) / peak) * 100 if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

            await sim_repo.create(
                state_date=state_date,
                bankroll=Decimal(str(round(bankroll, 2))),
                wins=wins,
                losses=losses,
                peak_bankroll=Decimal(str(round(peak, 2))),
                max_drawdown_pct=Decimal(str(round(max_dd, 4))),
                pick_id=None,
            )

        logger.info(f"Generated 30 days of mock data. Final bankroll: ${bankroll:.2f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="EdgeBet Backend CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-db command
    subparsers.add_parser("init-db", help="Initialize database and run migrations")

    # sync-teams command
    subparsers.add_parser("sync-teams", help="Sync NBA teams to database")

    # sync-historical-data command
    sync_parser = subparsers.add_parser("sync-historical-data", help="Sync historical games")
    sync_parser.add_argument(
        "--start-season", default="2022-23", help="Starting season (e.g., 2022-23)"
    )
    sync_parser.add_argument(
        "--end-season", default="2024-25", help="Ending season (e.g., 2024-25)"
    )

    # train-models command
    train_parser = subparsers.add_parser("train-models", help="Train ML models from Kaggle data")
    train_parser.add_argument(
        "--start-season", type=int, default=2020,
        help="Starting season year (e.g., 2020 for 2020-21)"
    )
    train_parser.add_argument(
        "--end-season", type=int, default=2024,
        help="Ending season year (e.g., 2024 for 2024-25)"
    )

    # sync-week-stats command
    week_parser = subparsers.add_parser(
        "sync-week-stats", help="Sync last N days of games with full box scores"
    )
    week_parser.add_argument(
        "--days", type=int, default=7, help="Number of days to sync (default: 7)"
    )

    # run-daily-job command
    subparsers.add_parser("run-daily-job", help="Manually run the daily job")

    # generate-mock-data command
    subparsers.add_parser("generate-mock-data", help="Generate mock simulation data")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Predict today's NBA games")
    predict_parser.add_argument(
        "--date", type=str, default=None,
        help="Date to predict (YYYY-MM-DD format, default: today)"
    )

    args = parser.parse_args()

    if args.command == "init-db":
        asyncio.run(init_db())
    elif args.command == "sync-teams":
        asyncio.run(sync_teams())
    elif args.command == "sync-historical-data":
        asyncio.run(sync_historical_data(args.start_season, args.end_season))
    elif args.command == "train-models":
        train_models(args.start_season, args.end_season)
    elif args.command == "sync-week-stats":
        asyncio.run(sync_week_stats(args.days))
    elif args.command == "run-daily-job":
        asyncio.run(run_daily_job_manual())
    elif args.command == "generate-mock-data":
        asyncio.run(generate_mock_data())
    elif args.command == "predict":
        from cli.predict import predict_today
        target_date = None
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        predict_today(target_date)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
