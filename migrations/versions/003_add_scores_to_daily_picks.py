"""Add home_score and away_score to daily_picks for caching game results

Revision ID: 003
Revises: 002
Create Date: 2026-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add score columns to cache game results
    op.add_column("daily_picks", sa.Column("home_score", sa.Integer(), nullable=True))
    op.add_column("daily_picks", sa.Column("away_score", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("daily_picks", "away_score")
    op.drop_column("daily_picks", "home_score")
