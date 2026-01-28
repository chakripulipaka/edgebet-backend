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


class PicksResponse(BaseModel):
    """Schema for the GET /picks response."""
    picks: List[PickResponse]
    allGamesComplete: bool = False  # True when all games for the day are final


class PicksQueryParams(BaseModel):
    """Query parameters for picks endpoint."""
    date: Optional[str] = None  # ISO date string
    bet_type: Optional[str] = None
    min_edge: Optional[float] = None
