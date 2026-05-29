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
    'Fulham': '#FFFFFF',
    'Ipswich': '#005DAA',
    'Leeds': '#FFCD00',
    'Leicester': '#0053A0',
    'Liverpool': '#C8102E',
    'Luton': '#F78F1E',
    'Man City': '#6CABDD',
    'Man Utd': '#DA291C',
    'Newcastle': '#241F20',
    "Nott'm Forest": '#DD0000',
    'Sheffield Utd': '#EE2737',
    'Southampton': '#D71920',
    'Spurs': '#132257',
    'Sunderland': '#EB172B',
    'West Ham': '#7A263A',
    'Wolves': '#FDB913'
}

FEATURES_GROUP = {

"target" : ["total_points"],

"id_cols" : ["name", "element", "kickoff_time", "gw", "code", "season"],

"static_cols" : ["position", "was_home", "starts"],

"fpl_cols" : ['bonus', 'bps', 'ict_index', 'influence', 'value', 'threat',  'selected', 'transfers_out', 'transfers_balance', 'transfers_in', "transfers_trend"],

"perf_cols" : [
 'yellow_cards',
 'xg_conceded_per_90',
 'clean_sheets',
 'minutes',
 'xg_involvements_per_90',
 'xg_per_90',
 'xa_per_90',
 'own_goals',
 'assists',
 'xg',
 'saves',
 'penalties_saved',
 'xp',
 'penalties_missed',
 'goals_scored',
 'team_h_score',
 'xa',
 'xg_involvements',
 'team_a_score',
 'goals_conceded',
 'creativity',
 'xg_conceded',
 'red_cards'],

"pre_game_cols" : ["name", "position", "element", "was_home", "was_home", "gw", "code", "season", "kickoff_time", "value", "selected", "transfers_in", "transfers_out", "transfers_balance", "transfers_trend"]
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
