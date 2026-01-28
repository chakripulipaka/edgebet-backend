from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.db.models import DailyPick


class PicksRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_date(self, pick_date: date) -> Optional[DailyPick]:
        """Get a single pick for a date (legacy method)."""
        result = await self.session.execute(
            select(DailyPick).where(DailyPick.pick_date == pick_date)
        )
        return result.scalar_one_or_none()

    async def get_all_by_date(self, pick_date: date) -> List[DailyPick]:
        """Get all picks for a specific date."""
        result = await self.session.execute(
            select(DailyPick)
            .where(DailyPick.pick_date == pick_date)
            .order_by(DailyPick.edge.desc())
        )
        return list(result.scalars().all())

    async def get_all_picks(self) -> List[DailyPick]:
        """Get all picks ordered by date descending."""
        result = await self.session.execute(
            select(DailyPick).order_by(DailyPick.pick_date.desc())
        )
        return list(result.scalars().all())

    async def get_picks_in_range(self, start_date: date, end_date: date) -> List[DailyPick]:
        """Get all picks within a date range."""
        result = await self.session.execute(
            select(DailyPick)
            .where(DailyPick.pick_date >= start_date, DailyPick.pick_date <= end_date)
            .order_by(DailyPick.pick_date, DailyPick.edge.desc())
        )
        return list(result.scalars().all())

    async def get_pending_picks(self) -> List[DailyPick]:
        """Get all picks that don't have an outcome yet."""
        result = await self.session.execute(
            select(DailyPick)
            .where(DailyPick.outcome.is_(None))
            .order_by(DailyPick.pick_date)
        )
        return list(result.scalars().all())

    async def create_pick(
        self,
        pick_date: date,
        espn_game_id: str,
        home_team: str,
        away_team: str,
        game_time: Optional[datetime],
        bet_type: str,
        side: str,
        odds: int,
        model_prob: Decimal,
        implied_prob: Decimal,
        edge: Decimal,
        kelly_capped: Decimal,
    ) -> DailyPick:
        """
        Create or update a pick (upsert based on unique constraint).

        If a pick already exists for the same (pick_date, espn_game_id, bet_type),
        it will be updated with the new values.
        """
        stmt = insert(DailyPick).values(
            pick_date=pick_date,
            espn_game_id=espn_game_id,
            home_team=home_team,
            away_team=away_team,
            game_time=game_time,
            bet_type=bet_type,
            side=side,
            odds=odds,
            model_prob=model_prob,
            implied_prob=implied_prob,
            edge=edge,
            kelly_capped=kelly_capped,
        )

        # On conflict, update the values (except outcome which should be preserved)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_pick_date_game_bet",
            set_={
                "home_team": home_team,
                "away_team": away_team,
                "game_time": game_time,
                "side": side,
                "odds": odds,
                "model_prob": model_prob,
                "implied_prob": implied_prob,
                "edge": edge,
                "kelly_capped": kelly_capped,
            }
        ).returning(DailyPick)

        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one()

    async def update_outcome(self, pick: DailyPick, outcome: str) -> DailyPick:
        """Update the outcome of a pick (win/loss)."""
        pick.outcome = outcome
        await self.session.commit()
        return pick

    async def update_outcome_with_scores(
        self,
        pick: DailyPick,
        home_score: int,
        away_score: int,
        outcome: str
    ) -> DailyPick:
        """Update the outcome and store game scores."""
        pick.home_score = home_score
        pick.away_score = away_score
        pick.outcome = outcome
        await self.session.commit()
        return pick

    async def delete(self, pick: DailyPick) -> None:
        """Delete a pick from the database."""
        await self.session.delete(pick)
        await self.session.commit()

    async def delete_picks_for_date_except(self, pick_date: date, keep_pick_id: int) -> int:
        """
        Delete all picks for a date except the specified one.
        Returns the number of picks deleted.
        """
        picks = await self.get_all_by_date(pick_date)
        deleted_count = 0
        for pick in picks:
            if pick.id != keep_pick_id:
                await self.session.delete(pick)
                deleted_count += 1
        await self.session.commit()
        return deleted_count

    async def get_picks_with_outcomes(self) -> List[DailyPick]:
        """Get all picks that have outcomes stored (for simulation)."""
        result = await self.session.execute(
            select(DailyPick)
            .where(DailyPick.outcome.isnot(None))
            .order_by(DailyPick.pick_date.asc())
        )
        return list(result.scalars().all())

    async def delete_all_for_date(self, pick_date: date) -> int:
        """
        Delete ALL picks for a date (for regeneration).

        Returns the number of picks deleted.
        """
        picks = await self.get_all_by_date(pick_date)
        count = len(picks)
        for pick in picks:
            await self.session.delete(pick)
        await self.session.commit()
        return count
