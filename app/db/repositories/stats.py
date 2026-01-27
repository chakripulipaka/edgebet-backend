from datetime import date
from typing import List, Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import TeamGameStats, Game


class StatsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_game_and_team(self, game_id: int, team_id: int) -> Optional[TeamGameStats]:
        result = await self.session.execute(
            select(TeamGameStats).where(
                TeamGameStats.game_id == game_id,
                TeamGameStats.team_id == team_id
            )
        )
        return result.scalar_one_or_none()

    async def get_team_rolling_stats(
        self, team_id: int, before_date: date, num_games: int = 10
    ) -> List[TeamGameStats]:
        """Get the last N games for a team before a given date."""
        result = await self.session.execute(
            select(TeamGameStats)
            .join(Game)
            .where(
                TeamGameStats.team_id == team_id,
                Game.game_date < before_date,
                Game.status == "final"
            )
            .order_by(Game.game_date.desc())
            .limit(num_games)
        )
        return list(result.scalars().all())

    async def create(
        self,
        game_id: int,
        team_id: int,
        is_home: bool,
        points: Optional[int] = None,
        fg_pct: Optional[float] = None,
        fg3_pct: Optional[float] = None,
        ft_pct: Optional[float] = None,
        assists: Optional[int] = None,
        rebounds: Optional[int] = None,
        blocks: Optional[int] = None,
        steals: Optional[int] = None,
        turnovers: Optional[int] = None,
        pace: Optional[float] = None,
        opponent_points: Optional[int] = None,
        rest_days: Optional[int] = None,
    ) -> TeamGameStats:
        stats = TeamGameStats(
            game_id=game_id,
            team_id=team_id,
            is_home=is_home,
            points=points,
            fg_pct=fg_pct,
            fg3_pct=fg3_pct,
            ft_pct=ft_pct,
            assists=assists,
            rebounds=rebounds,
            blocks=blocks,
            steals=steals,
            turnovers=turnovers,
            pace=pace,
            opponent_points=opponent_points,
            rest_days=rest_days,
        )
        self.session.add(stats)
        await self.session.commit()
        await self.session.refresh(stats)
        return stats

    async def upsert(self, **kwargs) -> TeamGameStats:
        game_id = kwargs["game_id"]
        team_id = kwargs["team_id"]
        existing = await self.get_by_game_and_team(game_id, team_id)
        if existing:
            for key, value in kwargs.items():
                if key not in ("game_id", "team_id") and value is not None:
                    setattr(existing, key, value)
            await self.session.commit()
            return existing
        return await self.create(**kwargs)
