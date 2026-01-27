"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Teams table
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nba_team_id", sa.Integer(), nullable=False),
        sa.Column("abbreviation", sa.String(3), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("city", sa.String(50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nba_team_id"),
    )
    op.create_index(op.f("ix_teams_id"), "teams", ["id"], unique=False)

    # Games table
    op.create_table(
        "games",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nba_game_id", sa.String(20), nullable=False),
        sa.Column("season", sa.String(10), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("game_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=True, server_default="scheduled"),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nba_game_id"),
    )
    op.create_index(op.f("ix_games_id"), "games", ["id"], unique=False)
    op.create_index(op.f("ix_games_game_date"), "games", ["game_date"], unique=False)

    # Team game stats table
    op.create_table(
        "team_game_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("is_home", sa.Boolean(), nullable=False),
        sa.Column("points", sa.Integer(), nullable=True),
        sa.Column("fg_pct", sa.Numeric(5, 3), nullable=True),
        sa.Column("fg3_pct", sa.Numeric(5, 3), nullable=True),
        sa.Column("ft_pct", sa.Numeric(5, 3), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("rebounds", sa.Integer(), nullable=True),
        sa.Column("blocks", sa.Integer(), nullable=True),
        sa.Column("steals", sa.Integer(), nullable=True),
        sa.Column("turnovers", sa.Integer(), nullable=True),
        sa.Column("pace", sa.Numeric(6, 2), nullable=True),
        sa.Column("opponent_points", sa.Integer(), nullable=True),
        sa.Column("rest_days", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "team_id", name="uq_game_team"),
    )
    op.create_index(op.f("ix_team_game_stats_id"), "team_game_stats", ["id"], unique=False)

    # Daily picks table
    op.create_table(
        "daily_picks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pick_date", sa.Date(), nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("bet_type", sa.String(20), nullable=False),
        sa.Column("side", sa.String(50), nullable=False),
        sa.Column("odds", sa.Integer(), nullable=False),
        sa.Column("model_prob", sa.Numeric(5, 4), nullable=False),
        sa.Column("implied_prob", sa.Numeric(5, 4), nullable=False),
        sa.Column("edge", sa.Numeric(6, 4), nullable=False),
        sa.Column("kelly_capped", sa.Numeric(5, 4), nullable=False),
        sa.Column("outcome", sa.String(10), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pick_date"),
    )
    op.create_index(op.f("ix_daily_picks_id"), "daily_picks", ["id"], unique=False)
    op.create_index(op.f("ix_daily_picks_pick_date"), "daily_picks", ["pick_date"], unique=False)

    # Simulation state table
    op.create_table(
        "simulation_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("state_date", sa.Date(), nullable=False),
        sa.Column("bankroll", sa.Numeric(12, 2), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("losses", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("peak_bankroll", sa.Numeric(12, 2), nullable=False),
        sa.Column("max_drawdown_pct", sa.Numeric(6, 4), nullable=True, server_default="0"),
        sa.Column("pick_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["pick_id"], ["daily_picks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("state_date"),
    )
    op.create_index(op.f("ix_simulation_state_id"), "simulation_state", ["id"], unique=False)
    op.create_index(op.f("ix_simulation_state_state_date"), "simulation_state", ["state_date"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_simulation_state_state_date"), table_name="simulation_state")
    op.drop_index(op.f("ix_simulation_state_id"), table_name="simulation_state")
    op.drop_table("simulation_state")

    op.drop_index(op.f("ix_daily_picks_pick_date"), table_name="daily_picks")
    op.drop_index(op.f("ix_daily_picks_id"), table_name="daily_picks")
    op.drop_table("daily_picks")

    op.drop_index(op.f("ix_team_game_stats_id"), table_name="team_game_stats")
    op.drop_table("team_game_stats")

    op.drop_index(op.f("ix_games_game_date"), table_name="games")
    op.drop_index(op.f("ix_games_id"), table_name="games")
    op.drop_table("games")

    op.drop_index(op.f("ix_teams_id"), table_name="teams")
    op.drop_table("teams")
