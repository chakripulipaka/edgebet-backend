"""
Simulation CLI - Backtest betting strategy using stored picks.

Usage:
    python -m cli.simulate --days 3 --bankroll 100
"""

import argparse
import asyncio
import re
import sys
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, ".")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import settings
from app.db.models import DailyPick
from app.services.espn_data import espn_data_service


def parse_spread(side: str) -> float:
    """
    Parse spread from side string.
    e.g., "Miami Heat -7.0" -> -7.0
    e.g., "Utah Jazz +3.5" -> 3.5
    """
    match = re.search(r'([+-]?\d+\.?\d*)\s*$', side)
    if match:
        return float(match.group(1))
    return 0.0


def parse_total_line(side: str) -> float:
    """
    Parse total line from side string.
    e.g., "Over 244.5" -> 244.5
    e.g., "Under 220" -> 220.0
    """
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

    Returns: (outcome, description)
        outcome: "win", "loss", or "push"
        description: Human-readable result
    """
    if pick.bet_type == "Moneyline":
        # Side is team name
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
        # Parse spread from side (e.g., "Miami Heat -7.0")
        spread = parse_spread(pick.side)
        picked_home = pick.side.startswith(pick.home_team)

        if picked_home:
            # Home team with spread
            adjusted_margin = (home_score - away_score) + spread
        else:
            # Away team with spread
            adjusted_margin = (away_score - home_score) + spread

        actual_margin = home_score - away_score
        if adjusted_margin > 0:
            return "win", f"Margin: {actual_margin:+d}, Spread: {spread:+.1f}"
        elif adjusted_margin < 0:
            return "loss", f"Margin: {actual_margin:+d}, Spread: {spread:+.1f}"
        else:
            return "push", f"Push - Margin: {actual_margin:+d}"

    elif pick.bet_type == "Total":
        # Parse total line (e.g., "Over 244.5" or "Under 220")
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


def calculate_payout(bet_amount: float, odds: int, outcome: str) -> float:
    """
    Calculate payout from American odds.

    Args:
        bet_amount: Amount wagered
        odds: American odds (e.g., -110, +150)
        outcome: "win", "loss", or "push"

    Returns:
        Profit/loss amount (negative for loss)
    """
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


async def get_picks_by_date(
    session: AsyncSession,
    target_date: date
) -> List[DailyPick]:
    """Get all picks for a specific date."""
    result = await session.execute(
        select(DailyPick)
        .where(DailyPick.pick_date == target_date)
        .order_by(DailyPick.edge.desc())
    )
    return list(result.scalars().all())


async def run_simulation(days: int, starting_bankroll: float):
    """
    Run backtest simulation using stored picks.

    Args:
        days: Number of days to simulate
        starting_bankroll: Starting portfolio value
    """
    # Create database connection
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    today = date.today()
    bankroll = starting_bankroll
    peak_bankroll = starting_bankroll
    max_drawdown_pct = 0.0
    wins = 0
    losses = 0
    pushes = 0

    print("\n" + "=" * 70)
    print("                       SIMULATION RESULTS")
    print("=" * 70)
    print(f" Starting Bankroll: ${starting_bankroll:.2f}")
    print(f" Days Simulated: {days}")
    print("=" * 70 + "\n")

    day_results = []

    async with async_session() as session:
        for day_offset in range(days, 0, -1):
            target_date = today - timedelta(days=day_offset)
            picks = await get_picks_by_date(session, target_date)

            if not picks:
                print(f"Day {days - day_offset + 1}: {target_date}")
                print(f"  No picks found for this date\n")
                continue

            # Get the pick with highest edge
            best_pick = picks[0]  # Already sorted by edge desc

            # Get actual game result from ESPN
            game_result = espn_data_service.get_game_result(best_pick.espn_game_id)

            if not game_result or game_result["status"] != "final":
                print(f"Day {days - day_offset + 1}: {target_date}")
                print(f"  Pick: {best_pick.side} ({best_pick.bet_type})")
                print(f"  Game not completed or result unavailable\n")
                continue

            home_score = game_result["home_score"]
            away_score = game_result["away_score"]

            if home_score is None or away_score is None:
                print(f"Day {days - day_offset + 1}: {target_date}")
                print(f"  Pick: {best_pick.side} ({best_pick.bet_type})")
                print(f"  Scores not available\n")
                continue

            # Determine outcome
            outcome, description = determine_outcome(best_pick, home_score, away_score)

            # Calculate bet size (Kelly)
            kelly_pct = float(best_pick.kelly_capped)
            bet_size = kelly_pct * bankroll

            # Calculate payout
            payout = calculate_payout(bet_size, best_pick.odds, outcome)

            # Update bankroll
            prev_bankroll = bankroll
            bankroll += payout

            # Track stats
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                pushes += 1

            # Track peak and drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

            # Print day result
            outcome_symbol = "✓" if outcome == "win" else ("✗" if outcome == "loss" else "—")
            print(f"Day {days - day_offset + 1}: {target_date}")
            print(f"  Pick: {best_pick.side} ({best_pick.bet_type}) {format_odds(best_pick.odds)} | Edge: {float(best_pick.edge):.1f}%")
            print(f"  Bet Size: ${bet_size:.2f} ({kelly_pct * 100:.1f}% Kelly)")
            print(f"  Result: {outcome.upper()} {outcome_symbol} | {description}")
            print(f"  Payout: {'+' if payout >= 0 else ''}{payout:.2f}")
            print(f"  Bankroll: ${bankroll:.2f}\n")

            day_results.append({
                "date": target_date,
                "pick": best_pick,
                "outcome": outcome,
                "payout": payout,
                "bankroll": bankroll,
            })

    # Print final summary
    total_bets = wins + losses + pushes
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    total_return = ((bankroll - starting_bankroll) / starting_bankroll) * 100

    print("=" * 70)
    print("                        FINAL RESULTS")
    print("=" * 70)
    print(f" Final Bankroll: ${bankroll:.2f}")
    print(f" Total Return: {'+' if total_return >= 0 else ''}{total_return:.2f}%")
    print(f" Record: {wins}-{losses}{f'-{pushes}' if pushes > 0 else ''} ({win_rate:.1f}%)")
    print(f" Peak Bankroll: ${peak_bankroll:.2f}")
    print(f" Max Drawdown: {max_drawdown_pct:.2f}%")
    print("=" * 70 + "\n")

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Run betting simulation backtest")
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to simulate (default: 3)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=100.0,
        help="Starting bankroll (default: 100)"
    )

    args = parser.parse_args()

    print(f"\nRunning simulation for last {args.days} days with ${args.bankroll:.2f} bankroll...")

    asyncio.run(run_simulation(args.days, args.bankroll))


if __name__ == "__main__":
    main()
