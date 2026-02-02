"""Add picks_snapshots table for pre-computed picks data

Revision ID: 005
Revises: 004
Create Date: 2026-02-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "picks_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("snapshot_data", JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_picks_snapshots_computed_at"),
        "picks_snapshots",
        ["computed_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_picks_snapshots_id"),
        "picks_snapshots",
        ["id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_picks_snapshots_id"), table_name="picks_snapshots")
    op.drop_index(op.f("ix_picks_snapshots_computed_at"), table_name="picks_snapshots")
    op.drop_table("picks_snapshots")
