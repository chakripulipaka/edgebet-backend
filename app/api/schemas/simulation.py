"""Pydantic schemas for simulation endpoints."""

from typing import List, Optional
from pydantic import BaseModel


class ChartDataPoint(BaseModel):
    """Single data point for the simulation chart."""
    day: int
    date: str
    bankroll: float


class BetHistoryEntry(BaseModel):
    """Single entry in bet history."""
    date: str
    pick: str
    betType: str
    homeTeam: str
    awayTeam: str
    odds: int
    modelProb: float
    impliedProb: float
    edge: float
    kellyBetSize: float
    kellyBetDollars: float
    portfolioBefore: float
    portfolioAfter: float
    outcome: str
    gameResult: str


class SimulationResponse(BaseModel):
    """Schema for the GET /simulation response."""
    chartData: List[ChartDataPoint]
    betHistory: List[BetHistoryEntry]
    finalBankroll: float
    wins: int
    losses: int
    pushes: int
    winRate: float
    roi: float
    maxDrawdown: float
    peakBankroll: float
    daysSimulated: int
    startingBankroll: float
