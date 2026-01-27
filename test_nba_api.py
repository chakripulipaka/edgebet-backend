"""Test script to debug NBA API box score issues."""

from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoretraditionalv3, scoreboardv2

# Test 1: Check what the scoreboard returns for a recent date
print('=== Scoreboard Test ===')
sb = scoreboardv2.ScoreboardV2(game_date='2026-01-17')
games = sb.game_header.get_data_frame()
print(f'Games found: {len(games)}')
if not games.empty:
    print(games[['GAME_ID', 'GAME_STATUS_ID', 'GAME_STATUS_TEXT']].head())

# Test 2: Try a specific game ID from the output
if not games.empty:
    game_id = games.iloc[0]['GAME_ID']

    # Try V2 endpoint
    print(f'\n=== Box Score V2 Test for {game_id} ===')
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        team_stats = box.team_stats.get_data_frame()
        print(f'V2 Team stats rows: {len(team_stats)}')
    except Exception as e:
        print(f'V2 Error: {e}')

    # Try V3 endpoint
    print(f'\n=== Box Score V3 Test for {game_id} ===')
    try:
        box3 = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        # V3 has different structure - check available attributes
        print(f'V3 data sets: {box3.get_dict().keys()}')
        data = box3.get_dict()
        if 'boxScoreTraditional' in data:
            bs = data['boxScoreTraditional']
            if 'homeTeam' in bs:
                home = bs['homeTeam']
                print(f"Home: {home.get('teamName')} - {home.get('statistics', {}).get('points', 'N/A')} pts")
            if 'awayTeam' in bs:
                away = bs['awayTeam']
                print(f"Away: {away.get('teamName')} - {away.get('statistics', {}).get('points', 'N/A')} pts")
    except Exception as e:
        print(f'V3 Error: {e}')
