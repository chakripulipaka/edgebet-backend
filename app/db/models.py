from sqlalchemy import (
    Column, Integer, String, Date, DateTime, Boolean,
    ForeignKey, Numeric, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship

from app.db.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    nba_team_id = Column(Integer, unique=True, nullable=False)
    abbreviation = Column(String(3), nullable=False)
    name = Column(String(100), nullable=False)
    city = Column(String(50), nullable=False)

    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    game_stats = relationship("TeamGameStats", back_populates="team")


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    nba_game_id = Column(String(20), unique=True, nullable=False)
    season = Column(String(10), nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    game_time = Column(DateTime(timezone=True), nullable=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status = Column(String(20), default="scheduled")

    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    team_stats = relationship("TeamGameStats", back_populates="game")


class TeamGameStats(Base):
    __tablename__ = "team_game_stats"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    is_home = Column(Boolean, nullable=False)
    points = Column(Integer, nullable=True)
    fg_pct = Column(Numeric(5, 3), nullable=True)
    fg3_pct = Column(Numeric(5, 3), nullable=True)
    ft_pct = Column(Numeric(5, 3), nullable=True)
    assists = Column(Integer, nullable=True)
    rebounds = Column(Integer, nullable=True)
    blocks = Column(Integer, nullable=True)
    steals = Column(Integer, nullable=True)
    turnovers = Column(Integer, nullable=True)
    pace = Column(Numeric(6, 2), nullable=True)
    opponent_points = Column(Integer, nullable=True)
    rest_days = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", name="uq_game_team"),
    )

    game = relationship("Game", back_populates="team_stats")
    team = relationship("Team", back_populates="game_stats")


class DailyPick(Base):
    __tablename__ = "daily_picks"

    id = Column(Integer, primary_key=True, index=True)
    pick_date = Column(Date, nullable=False, index=True)
    espn_game_id = Column(String(20), nullable=False)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    game_time = Column(DateTime(timezone=True), nullable=True)
    bet_type = Column(String(20), nullable=False)
    side = Column(String(50), nullable=False)
    odds = Column(Integer, nullable=False)
    model_prob = Column(Numeric(5, 4), nullable=False)
    implied_prob = Column(Numeric(5, 4), nullable=False)
    edge = Column(Numeric(6, 4), nullable=False)
    kelly_capped = Column(Numeric(5, 4), nullable=False)
    outcome = Column(String(10), nullable=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("pick_date", "espn_game_id", "bet_type", name="uq_pick_date_game_bet"),
    )

    simulation_states = relationship("SimulationState", back_populates="pick")


class SimulationState(Base):
    __tablename__ = "simulation_state"

    id = Column(Integer, primary_key=True, index=True)
    state_date = Column(Date, unique=True, nullable=False, index=True)
    bankroll = Column(Numeric(12, 2), nullable=False)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    peak_bankroll = Column(Numeric(12, 2), nullable=False)
    max_drawdown_pct = Column(Numeric(6, 4), default=0)
    pick_id = Column(Integer, ForeignKey("daily_picks.id"), nullable=True)

    pick = relationship("DailyPick", back_populates="simulation_states")


class SimulationSnapshot(Base):
    """Pre-computed simulation data snapshot, updated daily at 9 AM."""
    __tablename__ = "simulation_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    computed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    through_date = Column(Date, nullable=False)  # Data includes picks up to this date

    # Store the full SimulationResponse as JSON for fast retrieval
    # Contains: chartData, dailySummaries, finalBankroll, totalWins, totalLosses,
    # totalPushes, totalBets, winRate, roi, maxDrawdown, daysSimulated
    snapshot_data = Column(JSON, nullable=False)


class PicksSnapshot(Base):
    """Pre-computed picks data snapshot, updated hourly for instant dashboard loading."""
    __tablename__ = "picks_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    computed_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Store the full PicksResponse as JSON for fast retrieval
    # Contains: picks (array), gamesWithoutPicks (array), allGamesComplete (bool)
    snapshot_data = Column(JSON, nullable=False)
