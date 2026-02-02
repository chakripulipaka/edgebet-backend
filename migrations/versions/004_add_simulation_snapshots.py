"""Add simulation_snapshots table for pre-computed simulation data

Revision ID: 004
Revises: 003
Create Date: 2026-02-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "simulation_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("through_date", sa.Date(), nullable=False),
        sa.Column("snapshot_data", JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_simulation_snapshots_computed_at"),
        "simulation_snapshots",
        ["computed_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_simulation_snapshots_id"),
        "simulation_snapshots",
        ["id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_simulation_snapshots_id"), table_name="simulation_snapshots")
    op.drop_index(op.f("ix_simulation_snapshots_computed_at"), table_name="simulation_snapshots")
    op.drop_table("simulation_snapshots")
