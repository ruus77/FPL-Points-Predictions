import seaborn as sns
team_colors = {
    'Arsenal': '#EF0107',
    'Aston Villa': '#95BFE5',
    'Bournemouth': '#DA291C',
    'Brentford': '#E30613',
    'Brighton': '#0057B8',
    'Burnley': '#6C1D45',
    'Chelsea': '#034694',
    'Crystal Palace': '#1B458F',
    'Everton': '#003399',
    'Fulham': '#000000',
    'Ipswich': '#3A64A3',
    'Leeds': '#FFCD00',
    'Leicester': '#0053A0',
    'Liverpool': '#C8102E',
    'Man City': '#6CABDD',
    'Man Utd': '#DA291C',
    'Newcastle': '#241F20',
    "Nott'm Forest": '#DD0000',
    'Southampton': '#D71920',
    'Spurs': '#132257',
    'Sunderland': '#EB172B',
    'West Ham': '#7A263A',
    'Wolves': '#FDB913'
}

FEATURES_GROUP = {

"target" : ["total_points"],

"id_cols" : ["name", "element", "kickoff_time", "opponent_team", "gw", "code", "season", "player_id"],

"static_cols" : ["position", "was_home", "starts"],

"fpl_cols" : ['bonus', 'bps', 'ict_index', 'influence', 'value', 'threat', 'selected', 'transfers_balance', 'transfers_in', 'transfers_out', 'transfers_trend'],

"perf_cols" : [
    'xp', 'assists', 'clean_sheets', 'creativity', 'xa', 'xg_involvements', 'xg',
    'xg_conceded', 'goals_conceded', 'goals_scored', 'minutes', 'own_goals',
    'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 'yellow_cards',
    'team_score', 'opp_score', 'score_diff', 'shots', 'xg_chain', 'xg_buildup',
    'key_passes', 'team_points', 'opp_points', 'team_xp', 'opp_xp', 'team_goals',
    'opp_goals', 'team_xg', 'opp_xg', 'team_np_xg', 'opp_np_xg', 'team_np_xg_difference',
    'opp_np_xg_difference', 'team_ppda', 'opp_ppda', 'team_deep_completions',
    'opp_deep_completions', 'team_match_np_xg_diff', 'ppda_diff', 'xa_per_90',
    'xg_involvements_per_90', 'xg_per_90', 'xg_conceded_per_90', 'ict_index_per_90',
    'xg_chain_per_90', 'xg_buildup_per_90', 'team_xg_per_90', 'opp_xg_per_90',
    'team_np_xg_per_90', 'opp_np_xg_per_90', 'team_np_xg_difference_per_90',
    'opp_np_xg_difference_per_90', 'team_match_np_xg_diff_per_90', 'xg_share'
],
"pre_game_cols" : ["name", "position", "element", "was_home", "opponent_team", "was_home", "gw", "code", "season", "kickoff_time", "value", "selected", "transfers_in", "transfers_out", "transfers_balance", "transfers_trend"]
}


sns.set_theme(rc={
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})



players_x_stats = [
    'xg_per_90', 
    'xa_per_90', 
    'shots', 
    'key_passes', 
    'ict_index_per_90', 
    'xg_involvements_per_90'
]

teams_x_stats = [
    'team_xg_per_90', 
    'opp_xg_per_90', 
    'team_np_xg_difference_per_90', 
    'team_ppda', 
    'team_match_np_xg_diff_per_90'
]
