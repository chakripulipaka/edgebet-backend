"""Core betting calculations - must match frontend calculations exactly."""

from app.core.constants import MAX_KELLY


def calculate_implied_prob(odds: int) -> float:
    """
    Calculate implied probability from American odds.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability as a decimal (0.0 to 1.0)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def calculate_decimal_odds(odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 1.91, 2.50)
    """
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def calculate_kelly(model_prob: float, odds: int) -> float:
    """
    Calculate Kelly Criterion bet size, capped at MAX_KELLY (10%).

    Args:
        model_prob: Model's predicted probability (0.0 to 1.0)
        odds: American odds

    Returns:
        Kelly fraction capped at MAX_KELLY
    """
    decimal_odds = calculate_decimal_odds(odds)
    b = decimal_odds - 1  # Net odds (profit per unit wagered)
    kelly = (model_prob * b - (1 - model_prob)) / b
    return max(0, min(kelly, MAX_KELLY))


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """
    Calculate edge as the percentage difference between model and implied probability.

    Args:
        model_prob: Model's predicted probability (0.0 to 1.0)
        implied_prob: Implied probability from odds (0.0 to 1.0)

    Returns:
        Edge as a percentage (e.g., 5.2 for 5.2% edge)
    """
    return (model_prob - implied_prob) * 100


def calculate_bet_outcome(
    bet_type: str,
    side: str,
    home_score: int,
    away_score: int,
    home_team: str,
) -> str:
    """
    Determine if a bet won or lost based on final scores.

    Args:
        bet_type: "Moneyline", "Spread", or "Total"
        side: The side of the bet (e.g., "Lakers", "Lakers -4.5", "Over 220.5")
        home_score: Final home team score
        away_score: Final away team score
        home_team: Name of the home team

    Returns:
        "win" or "loss"
    """
    point_diff = home_score - away_score
    total_points = home_score + away_score

    if bet_type == "Moneyline":
        # Side is the team name
        home_won = point_diff > 0
        bet_on_home = side == home_team
        return "win" if home_won == bet_on_home else "loss"

    elif bet_type == "Spread":
        # Parse spread from side (e.g., "Lakers -4.5" or "Warriors +4.5")
        parts = side.rsplit(" ", 1)
        if len(parts) != 2:
            return "loss"  # Invalid format
        team_name, spread_str = parts
        try:
            spread = float(spread_str)
        except ValueError:
            return "loss"

        bet_on_home = team_name == home_team
        if bet_on_home:
            # Home team with spread (e.g., Lakers -4.5 means they need to win by more than 4.5)
            covered = point_diff + spread > 0
        else:
            # Away team with spread (e.g., Warriors +4.5 means they can lose by up to 4.5)
            covered = -point_diff + spread > 0
        return "win" if covered else "loss"

    elif bet_type == "Total":
        # Parse total from side (e.g., "Over 220.5" or "Under 220.5")
        parts = side.split(" ")
        if len(parts) != 2:
            return "loss"
        over_under, line_str = parts
        try:
            line = float(line_str)
        except ValueError:
            return "loss"

        if over_under == "Over":
            return "win" if total_points > line else "loss"
        else:  # Under
            return "win" if total_points < line else "loss"

    return "loss"


def calculate_payout(bet_amount: float, odds: int, won: bool) -> float:
    """
    Calculate the payout or loss from a bet.

    Args:
        bet_amount: Amount wagered
        odds: American odds
        won: Whether the bet won

    Returns:
        Net profit/loss (positive if won, negative if lost)
    """
    if not won:
        return -bet_amount

    decimal_odds = calculate_decimal_odds(odds)
    return bet_amount * (decimal_odds - 1)
