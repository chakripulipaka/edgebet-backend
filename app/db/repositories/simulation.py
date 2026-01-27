from datetime import date
from decimal import Decimal
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import SimulationState, DailyPick


class SimulationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_date(self, state_date: date) -> Optional[SimulationState]:
        result = await self.session.execute(
            select(SimulationState)
            .options(selectinload(SimulationState.pick))
            .where(SimulationState.state_date == state_date)
        )
        return result.scalar_one_or_none()

    async def get_latest(self) -> Optional[SimulationState]:
        result = await self.session.execute(
            select(SimulationState)
            .options(selectinload(SimulationState.pick))
            .order_by(SimulationState.state_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_all_states(self) -> List[SimulationState]:
        result = await self.session.execute(
            select(SimulationState)
            .options(selectinload(SimulationState.pick))
            .order_by(SimulationState.state_date)
        )
        return list(result.scalars().all())

    async def create(
        self,
        state_date: date,
        bankroll: Decimal,
        wins: int = 0,
        losses: int = 0,
        peak_bankroll: Optional[Decimal] = None,
        max_drawdown_pct: Decimal = Decimal("0"),
        pick_id: Optional[int] = None,
    ) -> SimulationState:
        state = SimulationState(
            state_date=state_date,
            bankroll=bankroll,
            wins=wins,
            losses=losses,
            peak_bankroll=peak_bankroll or bankroll,
            max_drawdown_pct=max_drawdown_pct,
            pick_id=pick_id,
        )
        self.session.add(state)
        await self.session.commit()
        await self.session.refresh(state)
        return state

    async def update(
        self,
        state: SimulationState,
        bankroll: Optional[Decimal] = None,
        wins: Optional[int] = None,
        losses: Optional[int] = None,
        peak_bankroll: Optional[Decimal] = None,
        max_drawdown_pct: Optional[Decimal] = None,
    ) -> SimulationState:
        if bankroll is not None:
            state.bankroll = bankroll
        if wins is not None:
            state.wins = wins
        if losses is not None:
            state.losses = losses
        if peak_bankroll is not None:
            state.peak_bankroll = peak_bankroll
        if max_drawdown_pct is not None:
            state.max_drawdown_pct = max_drawdown_pct
        await self.session.commit()
        return state
