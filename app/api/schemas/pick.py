"""Pydantic schemas for picks endpoints."""

from typing import Optional, List
from pydantic import BaseModel


class PickResponse(BaseModel):
    """Schema for a single pick in the response."""
    id: int
    homeTeam: str
    awayTeam: str
    gameTime: str
    betType: str
    side: str
    odds: int
    modelProb: float
    impliedProb: float
    edge: float
    player: Optional[str] = None
    gameStatus: str = "scheduled"  # scheduled, in_progress, halftime, final
    outcome: Optional[str] = None  # win, loss, push, or None if game not final
    espnGameId: Optional[str] = None  # For tracking in hourly job


class GameWithoutPicks(BaseModel):
    """Schema for a game that doesn't have picks yet (odds not available)."""
    homeTeam: str
    awayTeam: str
    gameTime: str
    gameStatus: str = "scheduled"
    espnGameId: Optional[str] = None


class PicksResponse(BaseModel):
    """Schema for the GET /picks response."""
    picks: List[PickResponse]
    gamesWithoutPicks: List[GameWithoutPicks] = []  # Games in window but no odds yet
    allGamesComplete: bool = False  # True when all games for the day are final
    computedAt: Optional[str] = None  # ISO timestamp of when snapshot was computed


class PicksQueryParams(BaseModel):
    """Query parameters for picks endpoint."""
    date: Optional[str] = None  # ISO date string
    bet_type: Optional[str] = None
    min_edge: Optional[float] = None
