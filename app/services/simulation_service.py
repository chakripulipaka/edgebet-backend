"""Service for managing the forward simulation and bankroll tracking."""

import logging
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.simulation import SimulationRepository
from app.db.repositories.picks import PicksRepository
from app.db.repositories.games import GameRepository
from app.db.models import SimulationState, DailyPick, Game
from app.core.constants import INITIAL_BANKROLL
from app.core.calculations import calculate_payout, calculate_kelly, calculate_bet_outcome
from app.services.espn_data import espn_data_service

logger = logging.getLogger(__name__)


def parse_spread(side: str) -> float:
    """Parse spread from side string. e.g., 'Miami Heat -7.0' -> -7.0"""
    match = re.search(r'([+-]?\d+\.?\d*)\s*$', side)
    if match:
        return float(match.group(1))
    return 0.0


def parse_total_line(side: str) -> float:
    """Parse total line from side string. e.g., 'Over 244.5' -> 244.5"""
    match = re.search(r'(\d+\.?\d*)\s*$', side)
    if match:
        return float(match.group(1))
    return 0.0


def determine_outcome(
    pick: DailyPick,
    home_score: int,
    away_score: int
) -> Tuple[str, str]:
    """
    Determine if a bet won based on actual scores.
    Returns: (outcome, description) - outcome is 'win', 'loss', or 'push'
    """
    if pick.bet_type == "Moneyline":
        picked_home = pick.side == pick.home_team
        home_won = home_score > away_score

        if picked_home:
            if home_won:
                return "win", f"{pick.home_team} {home_score}, {pick.away_team} {away_score}"
            else:
                return "loss", f"{pick.away_team} {away_score}, {pick.home_team} {home_score}"
        else:
            if not home_won:
                return "win", f"{pick.away_team} {away_score}, {pick.home_team} {home_score}"
            else:
                return "loss", f"{pick.home_team} {home_score}, {pick.away_team} {away_score}"

    elif pick.bet_type == "Spread":
        spread = parse_spread(pick.side)
        picked_home = pick.side.startswith(pick.home_team)

        if picked_home:
            adjusted_margin = (home_score - away_score) + spread
        else:
            adjusted_margin = (away_score - home_score) + spread

        actual_margin = home_score - away_score
        if adjusted_margin > 0:
            return "win", f"Margin: {actual_margin:+d}, Spread: {spread:+.1f}"
        elif adjusted_margin < 0:
            return "loss", f"Margin: {actual_margin:+d}, Spread: {spread:+.1f}"
        else:
            return "push", f"Push - Margin: {actual_margin:+d}"

    elif pick.bet_type == "Total":
        line = parse_total_line(pick.side)
        actual_total = home_score + away_score
        is_over = pick.side.startswith("Over")

        if is_over:
            if actual_total > line:
                return "win", f"Total: {actual_total}, Line: {line}"
            elif actual_total < line:
                return "loss", f"Total: {actual_total}, Line: {line}"
            else:
                return "push", f"Push - Total: {actual_total}"
        else:
            if actual_total < line:
                return "win", f"Total: {actual_total}, Line: {line}"
            elif actual_total > line:
                return "loss", f"Total: {actual_total}, Line: {line}"
            else:
                return "push", f"Push - Total: {actual_total}"

    return "unknown", "Could not determine outcome"


def calculate_payout_from_odds(bet_amount: float, odds: int, outcome: str) -> float:
    """Calculate payout from American odds."""
    if outcome == "loss":
        return -bet_amount
    elif outcome == "push":
        return 0.0
    elif outcome == "win":
        if odds > 0:
            return bet_amount * (odds / 100)
        else:
            return bet_amount * (100 / abs(odds))
    return 0.0


def format_odds(odds: int) -> str:
    """Format American odds with + sign for positive."""
    return f"+{odds}" if odds > 0 else str(odds)


class SimulationService:
    """Service for tracking simulation state and bankroll."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.sim_repo = SimulationRepository(session)
        self.picks_repo = PicksRepository(session)
        self.games_repo = GameRepository(session)

    async def get_simulation_data(self) -> dict:
        """
        Get full simulation data for the API response.

        Returns:
            Dictionary matching frontend simulation schema
        """
        states = await self.sim_repo.get_all_states()
        picks = await self.picks_repo.get_all_picks()

        if not states:
            # Return initial state if no simulation data
            return self._empty_simulation()

        # Build chart data
        chart_data = []
        for i, state in enumerate(states):
            chart_data.append({
                "day": i,
                "date": state.state_date.strftime("%b %d"),
                "bankroll": float(state.bankroll),
            })

        # Build bet history
        bet_history = []
        for pick in picks:
            if pick.outcome:  # Only include resolved picks
                state_before = await self._get_state_before_pick(pick)
                state_after = await self._get_state_after_pick(pick)

                bet_history.append({
                    "date": pick.pick_date.strftime("%b %d, %Y"),
                    "pick": self._format_pick_description(pick),
                    "odds": pick.odds,
                    "edge": float(pick.edge),
                    "kellyBetSize": float(pick.kelly_capped) * 100,
                    "kellyBetDollars": float(state_before.bankroll * pick.kelly_capped) if state_before else 0,
                    "portfolioBefore": float(state_before.bankroll) if state_before else INITIAL_BANKROLL,
                    "portfolioAfter": float(state_after.bankroll) if state_after else INITIAL_BANKROLL,
                    "outcome": pick.outcome,
                })

        # Calculate stats
        latest_state = states[-1] if states else None
        wins = latest_state.wins if latest_state else 0
        losses = latest_state.losses if latest_state else 0
        max_drawdown = float(latest_state.max_drawdown_pct) if latest_state else 0

        return {
            "chartData": chart_data,
            "finalBankroll": float(latest_state.bankroll) if latest_state else INITIAL_BANKROLL,
            "wins": wins,
            "losses": losses,
            "maxDrawdown": max_drawdown,
            "betHistory": bet_history,
        }

    async def _get_state_before_pick(self, pick: DailyPick) -> Optional[SimulationState]:
        """Get simulation state before a pick was resolved."""
        # State before is from the previous day
        prev_date = pick.pick_date - timedelta(days=1)
        return await self.sim_repo.get_by_date(prev_date)

    async def _get_state_after_pick(self, pick: DailyPick) -> Optional[SimulationState]:
        """Get simulation state after a pick was resolved."""
        return await self.sim_repo.get_by_date(pick.pick_date)

    def _format_pick_description(self, pick: DailyPick) -> str:
        """Format pick into a readable description."""
        if pick.bet_type == "Moneyline":
            return f"{pick.side} ML"
        elif pick.bet_type == "Spread":
            return pick.side
        elif pick.bet_type == "Total":
            return pick.side
        return pick.side

    def _empty_simulation(self) -> dict:
        """Return empty simulation data."""
        return {
            "chartData": [
                {"day": 0, "date": date.today().strftime("%b %d"), "bankroll": INITIAL_BANKROLL}
            ],
            "finalBankroll": INITIAL_BANKROLL,
            "wins": 0,
            "losses": 0,
            "maxDrawdown": 0,
            "betHistory": [],
        }

    async def initialize_simulation(self, start_date: date) -> SimulationState:
        """
        Initialize the simulation with starting bankroll.

        Args:
            start_date: Date to start simulation

        Returns:
            Initial SimulationState
        """
        existing = await self.sim_repo.get_by_date(start_date)
        if existing:
            return existing

        return await self.sim_repo.create(
            state_date=start_date,
            bankroll=Decimal(str(INITIAL_BANKROLL)),
            wins=0,
            losses=0,
            peak_bankroll=Decimal(str(INITIAL_BANKROLL)),
            max_drawdown_pct=Decimal("0"),
            pick_id=None,
        )

    async def update_simulation_for_pick(
        self,
        pick: DailyPick,
        outcome: str,
    ) -> SimulationState:
        """
        Update simulation state after a pick is resolved.

        Args:
            pick: The resolved pick
            outcome: "win" or "loss"

        Returns:
            Updated SimulationState
        """
        # Get previous state
        prev_state = await self.sim_repo.get_latest()
        if not prev_state:
            prev_state = await self.initialize_simulation(pick.pick_date - timedelta(days=1))

        prev_bankroll = float(prev_state.bankroll)
        kelly = float(pick.kelly_capped)
        bet_amount = prev_bankroll * kelly

        # Calculate new bankroll
        payout = calculate_payout(bet_amount, pick.odds, outcome == "win")
        new_bankroll = prev_bankroll + payout

        # Update wins/losses
        new_wins = prev_state.wins + (1 if outcome == "win" else 0)
        new_losses = prev_state.losses + (1 if outcome == "loss" else 0)

        # Track peak and drawdown
        new_peak = max(float(prev_state.peak_bankroll), new_bankroll)
        current_drawdown = ((new_peak - new_bankroll) / new_peak) * 100 if new_peak > 0 else 0
        max_drawdown = max(float(prev_state.max_drawdown_pct), current_drawdown)

        # Create new state
        return await self.sim_repo.create(
            state_date=pick.pick_date,
            bankroll=Decimal(str(round(new_bankroll, 2))),
            wins=new_wins,
            losses=new_losses,
            peak_bankroll=Decimal(str(round(new_peak, 2))),
            max_drawdown_pct=Decimal(str(round(max_drawdown, 4))),
            pick_id=pick.id,
        )

    async def resolve_pick(self, pick: DailyPick, game: Game) -> str:
        """
        Resolve a pick based on game results.

        Args:
            pick: Pick to resolve
            game: Game with final scores

        Returns:
            "win" or "loss"
        """
        if game.home_score is None or game.away_score is None:
            raise ValueError("Game not final yet")

        outcome = calculate_bet_outcome(
            pick.bet_type,
            pick.side,
            game.home_score,
            game.away_score,
            game.home_team.name,
        )

        # Update pick outcome
        await self.picks_repo.update_outcome(pick, outcome)

        # Update simulation
        await self.update_simulation_for_pick(pick, outcome)

        return outcome

    async def run_live_simulation(self, starting_bankroll: float = 100.0) -> dict:
        """
        Run simulation from stored picks with cached outcomes.

        Uses outcomes and scores stored in the database - NO ESPN API calls.
        Only includes picks that have been resolved (outcome is not None).

        Args:
            starting_bankroll: Starting portfolio value

        Returns:
            Dictionary with chartData, betHistory, and summary stats
        """
        # Get all picks with outcomes, ordered by date
        result = await self.session.execute(
            select(DailyPick)
            .where(DailyPick.outcome.isnot(None))
            .order_by(DailyPick.pick_date.asc())
        )
        resolved_picks = list(result.scalars().all())

        if not resolved_picks:
            return self._empty_live_simulation(starting_bankroll)

        bankroll = starting_bankroll
        peak_bankroll = starting_bankroll
        max_drawdown_pct = 0.0
        wins = 0
        losses = 0
        pushes = 0

        # Start chart with initial bankroll
        first_date = resolved_picks[0].pick_date
        chart_data = [{
            "day": 0,
            "date": first_date.strftime("%b %d"),
            "bankroll": starting_bankroll,
        }]
        bet_history = []
        day_counter = 1

        for pick in resolved_picks:
            # Use stored outcome - NO ESPN CALL
            outcome = pick.outcome

            if outcome == "unknown":
                continue

            # Calculate bet size (Kelly) based on current bankroll
            kelly_pct = float(pick.kelly_capped)
            bet_size = kelly_pct * bankroll

            # Calculate payout
            payout = calculate_payout_from_odds(bet_size, pick.odds, outcome)

            # Track stats
            portfolio_before = bankroll
            bankroll += payout
            portfolio_after = bankroll

            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                pushes += 1

            # Track peak and drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100 if peak_bankroll > 0 else 0
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

            # Add to chart data
            chart_data.append({
                "day": day_counter,
                "date": pick.pick_date.strftime("%b %d"),
                "bankroll": round(bankroll, 2),
            })
            day_counter += 1

            # Format game result description from stored scores
            description = self._format_game_result(pick)

            # Add to bet history
            bet_history.append({
                "date": pick.pick_date.strftime("%b %d, %Y"),
                "pick": pick.side,
                "betType": pick.bet_type,
                "homeTeam": pick.home_team,
                "awayTeam": pick.away_team,
                "odds": pick.odds,
                "modelProb": float(pick.model_prob),
                "impliedProb": float(pick.implied_prob),
                "edge": float(pick.edge),
                "kellyBetSize": kelly_pct * 100,
                "kellyBetDollars": round(bet_size, 2),
                "portfolioBefore": round(portfolio_before, 2),
                "portfolioAfter": round(portfolio_after, 2),
                "outcome": outcome,
                "gameResult": description,
            })

        total_bets = wins + losses + pushes
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        roi = ((bankroll - starting_bankroll) / starting_bankroll) * 100

        return {
            "chartData": chart_data,
            "betHistory": bet_history,
            "finalBankroll": round(bankroll, 2),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "winRate": round(win_rate, 1),
            "roi": round(roi, 2),
            "maxDrawdown": round(max_drawdown_pct, 2),
            "peakBankroll": round(peak_bankroll, 2),
            "daysSimulated": total_bets,
            "startingBankroll": starting_bankroll,
        }

    def _format_game_result(self, pick: DailyPick) -> str:
        """Format game result description from stored scores."""
        if pick.home_score is None or pick.away_score is None:
            return "Score not available"

        if pick.bet_type == "Moneyline":
            if pick.home_score > pick.away_score:
                return f"{pick.home_team} {pick.home_score}, {pick.away_team} {pick.away_score}"
            else:
                return f"{pick.away_team} {pick.away_score}, {pick.home_team} {pick.home_score}"
        elif pick.bet_type == "Spread":
            margin = pick.home_score - pick.away_score
            spread = parse_spread(pick.side)
            return f"Margin: {margin:+d}, Spread: {spread:+.1f}"
        elif pick.bet_type == "Total":
            total = pick.home_score + pick.away_score
            line = parse_total_line(pick.side)
            return f"Total: {total}, Line: {line}"
        return f"{pick.home_team} {pick.home_score} - {pick.away_team} {pick.away_score}"

    def _empty_live_simulation(self, starting_bankroll: float) -> dict:
        """Return empty simulation data for live simulation."""
        return {
            "chartData": [
                {"day": 0, "date": date.today().strftime("%b %d"), "bankroll": starting_bankroll}
            ],
            "betHistory": [],
            "finalBankroll": starting_bankroll,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "winRate": 0.0,
            "roi": 0.0,
            "maxDrawdown": 0.0,
            "peakBankroll": starting_bankroll,
            "daysSimulated": 0,
            "startingBankroll": starting_bankroll,
        }
