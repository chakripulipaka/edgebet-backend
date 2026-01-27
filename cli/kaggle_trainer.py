"""Standalone ML trainer using Kaggle SQLite dataset directly with pandas."""

import logging
import sqlite3
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from app.config import settings
from app.core.constants import FEATURE_NAMES, ROLLING_GAMES
from app.ml.models.moneyline import MoneylineModel
from app.ml.models.spread import SpreadModel
from app.ml.models.totals import TotalsModel

logger = logging.getLogger(__name__)


def download_kaggle_dataset() -> str:
    """
    Download the Kaggle NBA dataset.

    Returns:
        Path to the downloaded SQLite database
    """
    logger.info("Downloading Kaggle NBA dataset...")
    dataset_path = kagglehub.dataset_download("wyattowalsh/basketball")
    db_path = Path(dataset_path) / "nba.sqlite"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    logger.info(f"Dataset downloaded to {db_path}")
    return str(db_path)


def load_games_from_kaggle(
    db_path: str,
    start_season: int = 2020,
    end_season: int = 2024
) -> pd.DataFrame:
    """
    Load games from Kaggle SQLite database.

    Args:
        db_path: Path to SQLite database
        start_season: Starting season year (e.g., 2020 for 2020-21)
        end_season: Ending season year (e.g., 2024 for 2024-25)

    Returns:
        DataFrame with game data
    """
    conn = sqlite3.connect(db_path)

    # Build season filter for regular season games (prefix 2)
    season_ids = [f"2{year}" for year in range(start_season, end_season + 1)]
    season_filter = ",".join(f"'{s}'" for s in season_ids)

    query = f"""
    SELECT
        game_id,
        game_date,
        season_id,
        team_id_home,
        team_id_away,
        team_abbreviation_home,
        team_abbreviation_away,
        pts_home,
        pts_away,
        fg_pct_home,
        fg_pct_away,
        fg3_pct_home,
        fg3_pct_away,
        ft_pct_home,
        ft_pct_away,
        ast_home,
        ast_away,
        reb_home,
        reb_away,
        blk_home,
        blk_away,
        stl_home,
        stl_away,
        tov_home,
        tov_away
    FROM game
    WHERE season_id IN ({season_filter})
      AND pts_home IS NOT NULL
      AND pts_away IS NOT NULL
    ORDER BY game_date
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])

    logger.info(f"Loaded {len(df)} games from Kaggle dataset")
    return df


def compute_rolling_stats(df: pd.DataFrame, rolling_games: int = ROLLING_GAMES) -> pd.DataFrame:
    """
    Compute rolling statistics for each team.

    Creates rolling averages for each team based on their last N games,
    computed separately for home and away games to get team-level stats.

    Args:
        df: DataFrame with game data
        rolling_games: Number of games for rolling window

    Returns:
        DataFrame with rolling stats added
    """
    # Stats columns we care about
    stat_cols = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'blk', 'stl', 'tov']

    # Create team-centric view: Each row represents one team's performance in one game
    home_games = df[['game_id', 'game_date', 'team_id_home', 'team_id_away'] +
                    [f'{col}_home' for col in stat_cols] +
                    [f'{col}_away' for col in stat_cols]].copy()
    home_games = home_games.rename(columns={
        'team_id_home': 'team_id',
        'team_id_away': 'opponent_id',
        **{f'{col}_home': col for col in stat_cols},
        **{f'{col}_away': f'opp_{col}' for col in stat_cols}
    })
    home_games['is_home'] = True

    away_games = df[['game_id', 'game_date', 'team_id_away', 'team_id_home'] +
                    [f'{col}_away' for col in stat_cols] +
                    [f'{col}_home' for col in stat_cols]].copy()
    away_games = away_games.rename(columns={
        'team_id_away': 'team_id',
        'team_id_home': 'opponent_id',
        **{f'{col}_away': col for col in stat_cols},
        **{f'{col}_home': f'opp_{col}' for col in stat_cols}
    })
    away_games['is_home'] = False

    # Combine into single team-game view
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(['team_id', 'game_date'])

    # Compute rolling averages per team (shift to exclude current game)
    rolling_cols = stat_cols + [f'opp_{col}' for col in stat_cols if col == 'pts']

    for col in rolling_cols:
        team_games[f'rolling_{col}'] = team_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=rolling_games, min_periods=5).mean()
        )

    # Calculate rest days
    team_games['prev_game_date'] = team_games.groupby('team_id')['game_date'].shift(1)
    team_games['rest_days'] = (team_games['game_date'] - team_games['prev_game_date']).dt.days - 1
    team_games['rest_days'] = team_games['rest_days'].clip(lower=0, upper=7).fillna(3)

    # Compute rolling rest days average
    team_games['rolling_rest_days'] = team_games.groupby('team_id')['rest_days'].transform(
        lambda x: x.shift(1).rolling(window=rolling_games, min_periods=5).mean()
    )

    return team_games


def build_training_features(df: pd.DataFrame, team_rolling_stats: pd.DataFrame) -> tuple:
    """
    Build feature matrix and target variables from games and rolling stats.

    Args:
        df: Original games DataFrame
        team_rolling_stats: DataFrame with rolling stats per team-game

    Returns:
        Tuple of (X, y_moneyline, y_spread, y_total)
    """
    X_list = []
    y_ml_list = []
    y_spread_list = []
    y_total_list = []

    # Create lookup for rolling stats: (team_id, game_id) -> stats
    home_stats_lookup = team_rolling_stats[team_rolling_stats['is_home'] == True].set_index(
        ['team_id', 'game_id']
    )
    away_stats_lookup = team_rolling_stats[team_rolling_stats['is_home'] == False].set_index(
        ['team_id', 'game_id']
    )

    for _, row in df.iterrows():
        game_id = row['game_id']
        home_team_id = row['team_id_home']
        away_team_id = row['team_id_away']

        # Get rolling stats for both teams
        try:
            home_rolling = home_stats_lookup.loc[(home_team_id, game_id)]
            away_rolling = away_stats_lookup.loc[(away_team_id, game_id)]
        except KeyError:
            continue

        # Check if we have sufficient data (rolling stats are not NaN)
        if pd.isna(home_rolling['rolling_pts']) or pd.isna(away_rolling['rolling_pts']):
            continue

        # Build feature vector (13 features matching FEATURE_NAMES)
        features = np.array([
            home_rolling['rolling_pts'] - away_rolling['rolling_pts'],           # points_diff
            home_rolling['rolling_opp_pts'] - away_rolling['rolling_opp_pts'],   # opponent_points_diff
            home_rolling['rolling_fg_pct'] - away_rolling['rolling_fg_pct'],     # fg_pct_diff
            home_rolling['rolling_fg3_pct'] - away_rolling['rolling_fg3_pct'],   # fg3_pct_diff
            home_rolling['rolling_ft_pct'] - away_rolling['rolling_ft_pct'],     # ft_pct_diff
            home_rolling['rolling_ast'] - away_rolling['rolling_ast'],           # assists_diff
            home_rolling['rolling_reb'] - away_rolling['rolling_reb'],           # rebounds_diff
            home_rolling['rolling_blk'] - away_rolling['rolling_blk'],           # blocks_diff
            home_rolling['rolling_stl'] - away_rolling['rolling_stl'],           # steals_diff
            home_rolling['rolling_tov'] - away_rolling['rolling_tov'],           # turnovers_diff
            0.0,  # pace_diff (not available in Kaggle data, use 0)
            home_rolling['rolling_rest_days'] - away_rolling['rolling_rest_days'],  # rest_days_diff
            1.0,  # is_home (always 1)
        ])

        X_list.append(features)
        y_ml_list.append(1 if row['pts_home'] > row['pts_away'] else 0)
        y_spread_list.append(row['pts_home'] - row['pts_away'])
        y_total_list.append(row['pts_home'] + row['pts_away'])

    if not X_list:
        return None, None, None, None

    return (
        np.array(X_list),
        np.array(y_ml_list),
        np.array(y_spread_list),
        np.array(y_total_list),
    )


def train_from_kaggle(start_season: int = 2020, end_season: int = 2024) -> dict:
    """
    Train all ML models directly from Kaggle SQLite data.

    Args:
        start_season: Starting season year (e.g., 2020 for 2020-21)
        end_season: Ending season year (e.g., 2024 for 2024-25)

    Returns:
        Dictionary with training results for all models
    """
    # Step 1: Download/cache Kaggle dataset
    db_path = download_kaggle_dataset()

    # Step 2: Load games into pandas
    df = load_games_from_kaggle(db_path, start_season, end_season)

    if df.empty:
        logger.error("No games found in specified date range")
        return {"error": "No games found"}

    logger.info(f"Processing {len(df)} games for training...")

    # Step 3: Compute rolling stats per team
    team_rolling_stats = compute_rolling_stats(df)

    # Step 4: Build feature matrix
    X, y_ml, y_spread, y_total = build_training_features(df, team_rolling_stats)

    if X is None or len(X) == 0:
        logger.error("No training data generated (insufficient rolling history)")
        return {"error": "No training data generated"}

    logger.info(f"Generated {len(X)} training samples with {X.shape[1]} features")

    # Ensure model directory exists
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Step 5: Train Moneyline model
    logger.info("Training moneyline model...")
    ml_model = MoneylineModel()
    results["moneyline"] = ml_model.train(X, y_ml)
    ml_model.save(str(model_dir / "moneyline.joblib"))
    logger.info(f"Moneyline CV accuracy: {results['moneyline']['cv_accuracy_mean']:.3f}")

    # Train Spread model
    logger.info("Training spread model...")
    spread_model = SpreadModel()
    results["spread"] = spread_model.train(X, y_spread)
    spread_model.save(str(model_dir / "spread.joblib"))
    logger.info(f"Spread CV MAE: {results['spread']['cv_mae_mean']:.2f}")

    # Train Totals model
    logger.info("Training totals model...")
    totals_model = TotalsModel()
    results["totals"] = totals_model.train(X, y_total)
    totals_model.save(str(model_dir / "totals.joblib"))
    logger.info(f"Totals CV MAE: {results['totals']['cv_mae_mean']:.2f}")

    # Step 6: Print summary
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"  Seasons: {start_season}-{end_season}")
    logger.info(f"  Total games: {len(df)}")
    logger.info(f"  Training samples: {len(X)}")
    logger.info(f"  Home win rate: {results['moneyline']['home_win_rate']:.1%}")
    logger.info(f"  Models saved to: {model_dir}")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_from_kaggle()
