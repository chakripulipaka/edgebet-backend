"""Constants and team mappings for the EdgeBet platform."""

# NBA Team ID to abbreviation mapping
NBA_TEAM_MAP = {
    1610612737: {"abbreviation": "ATL", "name": "Hawks", "city": "Atlanta"},
    1610612738: {"abbreviation": "BOS", "name": "Celtics", "city": "Boston"},
    1610612751: {"abbreviation": "BKN", "name": "Nets", "city": "Brooklyn"},
    1610612766: {"abbreviation": "CHA", "name": "Hornets", "city": "Charlotte"},
    1610612741: {"abbreviation": "CHI", "name": "Bulls", "city": "Chicago"},
    1610612739: {"abbreviation": "CLE", "name": "Cavaliers", "city": "Cleveland"},
    1610612742: {"abbreviation": "DAL", "name": "Mavericks", "city": "Dallas"},
    1610612743: {"abbreviation": "DEN", "name": "Nuggets", "city": "Denver"},
    1610612765: {"abbreviation": "DET", "name": "Pistons", "city": "Detroit"},
    1610612744: {"abbreviation": "GSW", "name": "Warriors", "city": "Golden State"},
    1610612745: {"abbreviation": "HOU", "name": "Rockets", "city": "Houston"},
    1610612754: {"abbreviation": "IND", "name": "Pacers", "city": "Indiana"},
    1610612746: {"abbreviation": "LAC", "name": "Clippers", "city": "Los Angeles"},
    1610612747: {"abbreviation": "LAL", "name": "Lakers", "city": "Los Angeles"},
    1610612763: {"abbreviation": "MEM", "name": "Grizzlies", "city": "Memphis"},
    1610612748: {"abbreviation": "MIA", "name": "Heat", "city": "Miami"},
    1610612749: {"abbreviation": "MIL", "name": "Bucks", "city": "Milwaukee"},
    1610612750: {"abbreviation": "MIN", "name": "Timberwolves", "city": "Minnesota"},
    1610612740: {"abbreviation": "NOP", "name": "Pelicans", "city": "New Orleans"},
    1610612752: {"abbreviation": "NYK", "name": "Knicks", "city": "New York"},
    1610612760: {"abbreviation": "OKC", "name": "Thunder", "city": "Oklahoma City"},
    1610612753: {"abbreviation": "ORL", "name": "Magic", "city": "Orlando"},
    1610612755: {"abbreviation": "PHI", "name": "76ers", "city": "Philadelphia"},
    1610612756: {"abbreviation": "PHX", "name": "Suns", "city": "Phoenix"},
    1610612757: {"abbreviation": "POR", "name": "Trail Blazers", "city": "Portland"},
    1610612758: {"abbreviation": "SAC", "name": "Kings", "city": "Sacramento"},
    1610612759: {"abbreviation": "SAS", "name": "Spurs", "city": "San Antonio"},
    1610612761: {"abbreviation": "TOR", "name": "Raptors", "city": "Toronto"},
    1610612762: {"abbreviation": "UTA", "name": "Jazz", "city": "Utah"},
    1610612764: {"abbreviation": "WAS", "name": "Wizards", "city": "Washington"},
}

# Abbreviation to full team name
ABBREV_TO_NAME = {v["abbreviation"]: v["name"] for v in NBA_TEAM_MAP.values()}

# ESPN Team ID to NBA API Team ID mapping
ESPN_TO_NBA_TEAM_MAP = {
    1: 1610612737,   # ATL Hawks
    2: 1610612738,   # BOS Celtics
    17: 1610612751,  # BKN Nets
    30: 1610612766,  # CHA Hornets
    4: 1610612741,   # CHI Bulls
    5: 1610612739,   # CLE Cavaliers
    6: 1610612742,   # DAL Mavericks
    7: 1610612743,   # DEN Nuggets
    8: 1610612765,   # DET Pistons
    9: 1610612744,   # GSW Warriors
    10: 1610612745,  # HOU Rockets
    11: 1610612754,  # IND Pacers
    12: 1610612746,  # LAC Clippers
    13: 1610612747,  # LAL Lakers
    29: 1610612763,  # MEM Grizzlies
    14: 1610612748,  # MIA Heat
    15: 1610612749,  # MIL Bucks
    16: 1610612750,  # MIN Timberwolves
    3: 1610612740,   # NOP Pelicans
    18: 1610612752,  # NYK Knicks
    25: 1610612760,  # OKC Thunder
    19: 1610612753,  # ORL Magic
    20: 1610612755,  # PHI 76ers
    21: 1610612756,  # PHX Suns
    22: 1610612757,  # POR Trail Blazers
    23: 1610612758,  # SAC Kings
    24: 1610612759,  # SAS Spurs
    28: 1610612761,  # TOR Raptors
    26: 1610612762,  # UTA Jazz
    27: 1610612764,  # WAS Wizards
}

# Reverse mapping: NBA API Team ID to ESPN Team ID
NBA_TO_ESPN_TEAM_MAP = {v: k for k, v in ESPN_TO_NBA_TEAM_MAP.items()}

# Bet types
BET_TYPES = ["Moneyline", "Spread", "Total"]

# Feature names for ML models (13 total)
FEATURE_NAMES = [
    "points_diff",
    "opponent_points_diff",
    "fg_pct_diff",
    "fg3_pct_diff",
    "ft_pct_diff",
    "assists_diff",
    "rebounds_diff",
    "blocks_diff",
    "steals_diff",
    "turnovers_diff",
    "pace_diff",
    "rest_days_diff",
    "is_home",
]

# Rolling games count for feature calculation
ROLLING_GAMES = 10

# Initial bankroll for simulation
INITIAL_BANKROLL = 100.0

# Maximum Kelly fraction (10% cap)
MAX_KELLY = 0.10
