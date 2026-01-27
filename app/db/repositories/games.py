from datetime import date
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Game, Team


class GameRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_nba_id(self, nba_game_id: str) -> Optional[Game]:
        result = await self.session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.nba_game_id == nba_game_id)
        )
        return result.scalar_one_or_none()

    async def get_by_date(self, game_date: date) -> List[Game]:
        result = await self.session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.game_date == game_date)
            .order_by(Game.game_time)
        )
        return list(result.scalars().all())

    async def get_scheduled_by_date(self, game_date: date) -> List[Game]:
        result = await self.session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.game_date == game_date, Game.status == "scheduled")
            .order_by(Game.game_time)
        )
        return list(result.scalars().all())

    async def get_games_in_range(self, start_date: date, end_date: date) -> List[Game]:
        result = await self.session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.game_date >= start_date, Game.game_date <= end_date)
            .order_by(Game.game_date, Game.game_time)
        )
        return list(result.scalars().all())

    async def get_final_games(self, start_date: date, end_date: date) -> List[Game]:
        result = await self.session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(
                Game.game_date >= start_date,
                Game.game_date <= end_date,
                Game.status == "final"
            )
            .order_by(Game.game_date)
        )
        return list(result.scalars().all())

    async def create(
        self,
        nba_game_id: str,
        season: str,
        game_date: date,
        game_time,
        home_team_id: int,
        away_team_id: int,
        status: str = "scheduled",
    ) -> Game:
        game = Game(
            nba_game_id=nba_game_id,
            season=season,
            game_date=game_date,
            game_time=game_time,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            status=status,
        )
        self.session.add(game)
        await self.session.commit()
        await self.session.refresh(game)
        return game

    async def update_score(
        self, game: Game, home_score: int, away_score: int, status: str = "final"
    ) -> Game:
        game.home_score = home_score
        game.away_score = away_score
        game.status = status
        await self.session.commit()
        return game
