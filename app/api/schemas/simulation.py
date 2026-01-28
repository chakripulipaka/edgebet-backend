"""Pydantic schemas for simulation endpoints."""

from typing import List, Optional
from pydantic import BaseModel


class ChartDataPoint(BaseModel):
    """Single data point for the simulation chart."""
    day: int
    date: str
    bankroll: float


class BetDetail(BaseModel):
    """Individual bet within a day."""
    game: str
    pick: str
    betType: str
    odds: int
    modelProb: float  # As percentage (e.g., 62.5)
    impliedProb: float  # As percentage (e.g., 52.4)
    edge: float  # As percentage (e.g., 10.1)
    kellyDollars: float  # Bet amount in dollars
    outcome: str  # "win", "loss", or "push"
    payout: float  # Net payout (positive for win, negative for loss)


class DailySummary(BaseModel):
    """Summary for a single day's betting."""
    date: str
    portfolioBefore: float
    portfolioAfter: float
    netProfitLoss: float
    record: str  # e.g., "3-2" or "3-2-1" if pushes
    bets: List[BetDetail]


class SimulationResponse(BaseModel):
    """Schema for the GET /simulation response."""
    chartData: List[ChartDataPoint]
    dailySummaries: List[DailySummary]
    finalBankroll: float
    totalWins: int
    totalLosses: int
    totalPushes: int
    winRate: float
    roi: float
    maxDrawdown: float
    peakBankroll: float
    daysSimulated: int
    totalBets: int
    startingBankroll: float
