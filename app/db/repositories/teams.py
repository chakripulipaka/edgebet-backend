from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Team


class TeamRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_nba_id(self, nba_team_id: int) -> Optional[Team]:
        result = await self.session.execute(
            select(Team).where(Team.nba_team_id == nba_team_id)
        )
        return result.scalar_one_or_none()

    async def get_by_abbreviation(self, abbreviation: str) -> Optional[Team]:
        result = await self.session.execute(
            select(Team).where(Team.abbreviation == abbreviation.upper())
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> List[Team]:
        result = await self.session.execute(select(Team))
        return list(result.scalars().all())

    async def create(self, nba_team_id: int, abbreviation: str, name: str, city: str) -> Team:
        team = Team(
            nba_team_id=nba_team_id,
            abbreviation=abbreviation,
            name=name,
            city=city,
        )
        self.session.add(team)
        await self.session.commit()
        await self.session.refresh(team)
        return team

    async def upsert(self, nba_team_id: int, abbreviation: str, name: str, city: str) -> Team:
        existing = await self.get_by_nba_id(nba_team_id)
        if existing:
            existing.abbreviation = abbreviation
            existing.name = name
            existing.city = city
            await self.session.commit()
            return existing
        return await self.create(nba_team_id, abbreviation, name, city)
