"""Allow multiple picks per day with denormalized game info

Revision ID: 002
Revises: 001
Create Date: 2026-01-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the foreign key constraint from daily_picks to games
    op.drop_constraint("daily_picks_game_id_fkey", "daily_picks", type_="foreignkey")

    # Drop the unique constraint on pick_date (allows only 1 pick per day)
    op.drop_constraint("daily_picks_pick_date_key", "daily_picks", type_="unique")

    # Add new columns for denormalized game info
    op.add_column("daily_picks", sa.Column("espn_game_id", sa.String(20), nullable=True))
    op.add_column("daily_picks", sa.Column("home_team", sa.String(100), nullable=True))
    op.add_column("daily_picks", sa.Column("away_team", sa.String(100), nullable=True))
    op.add_column("daily_picks", sa.Column("game_time", sa.DateTime(timezone=True), nullable=True))

    # Migrate existing data: copy game info from games table
    op.execute("""
        UPDATE daily_picks dp
        SET
            espn_game_id = g.nba_game_id,
            home_team = CONCAT(t_home.city, ' ', t_home.name),
            away_team = CONCAT(t_away.city, ' ', t_away.name),
            game_time = g.game_time
        FROM games g
        JOIN teams t_home ON g.home_team_id = t_home.id
        JOIN teams t_away ON g.away_team_id = t_away.id
        WHERE dp.game_id = g.id
    """)

    # Set default values for any rows that couldn't be migrated
    op.execute("""
        UPDATE daily_picks
        SET espn_game_id = 'unknown'
        WHERE espn_game_id IS NULL
    """)
    op.execute("""
        UPDATE daily_picks
        SET home_team = 'Unknown Home'
        WHERE home_team IS NULL
    """)
    op.execute("""
        UPDATE daily_picks
        SET away_team = 'Unknown Away'
        WHERE away_team IS NULL
    """)

    # Make the new columns non-nullable
    op.alter_column("daily_picks", "espn_game_id", nullable=False)
    op.alter_column("daily_picks", "home_team", nullable=False)
    op.alter_column("daily_picks", "away_team", nullable=False)

    # Drop the old game_id column
    op.drop_column("daily_picks", "game_id")

    # Add new unique constraint for (pick_date, espn_game_id, bet_type)
    op.create_unique_constraint(
        "uq_pick_date_game_bet",
        "daily_picks",
        ["pick_date", "espn_game_id", "bet_type"]
    )


def downgrade() -> None:
    # Drop the new unique constraint
    op.drop_constraint("uq_pick_date_game_bet", "daily_picks", type_="unique")

    # Add back the game_id column
    op.add_column("daily_picks", sa.Column("game_id", sa.Integer(), nullable=True))

    # Try to restore game_id from espn_game_id
    op.execute("""
        UPDATE daily_picks dp
        SET game_id = g.id
        FROM games g
        WHERE dp.espn_game_id = g.nba_game_id
    """)

    # Make game_id non-nullable (this may fail if there are picks without matching games)
    op.alter_column("daily_picks", "game_id", nullable=False)

    # Drop the denormalized columns
    op.drop_column("daily_picks", "game_time")
    op.drop_column("daily_picks", "away_team")
    op.drop_column("daily_picks", "home_team")
    op.drop_column("daily_picks", "espn_game_id")

    # Re-add the unique constraint on pick_date
    op.create_unique_constraint("daily_picks_pick_date_key", "daily_picks", ["pick_date"])

    # Re-add the foreign key constraint
    op.create_foreign_key(
        "daily_picks_game_id_fkey",
        "daily_picks",
        "games",
        ["game_id"],
        ["id"]
    )
