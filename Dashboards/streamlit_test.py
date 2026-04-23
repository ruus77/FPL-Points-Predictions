import pandas as pd
import streamlit as st

import sys
from pathlib import Path
import importlib.util

root_path = Path(__file__).resolve().parent.parent

config_file_path = root_path / "config.py"
spec = importlib.util.spec_from_file_location("config", config_file_path)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

scripts_path = root_path / "Scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

from data_utils.visualization import radar_plot_x_stats
from data_utils.colors_config import players_x_stats, teams_x_stats


@st.cache_data
def load_data():
    df = pd.read_parquet(config.TIDY_DATA_PATH)
    df_ema = pd.read_parquet(config.FPL_DATA_PATH)
    preds = pd.read_parquet(config.PREDICTED_STATS_PATH)

    df.columns = df.columns.str.strip()
    df_ema.columns = df_ema.columns.str.strip()
    preds.columns = preds.columns.str.strip()
    return df, preds, df_ema

data, preds, data_ema = load_data()

curr_season = data.season.sort_values().unique()[-1]

@st.cache_data
def get_players(season):
    return data[data.season == season].name.sort_values().unique()

players = get_players(curr_season)

with st.sidebar:
    st.title("Dashboard Settings")
    selected_player = st.multiselect(
        "Select players to compare",
        players,
        default=[]
    )

st.title("FPL Points Predictions Dashboard")

if selected_player:
    st.header("Player Comparison")

    st.subheader("Current Prices")
    cols = st.columns(len(selected_player))

    for i, p in enumerate(selected_player):
        p_data = data[(data.name == p) & (data.season == curr_season)].sort_values(by="gw")

        if not p_data.empty:
            latest_price = p_data.iloc[-1]["value"]
            position = p_data.position.unique()
            with cols[i]:
                st.metric(label=f"{p}:   {position[0]}", value=f"£{latest_price:.1f}m")
        else:
            with cols[i]:
                st.write(f"{p}: No data available")

    st.divider()
    player_season_data = data[(data.name.isin(selected_player)) & (data.season == curr_season)]
    player_season_data = player_season_data.sort_values("gw")

    st.subheader("Points History")
    st.line_chart(
        data=player_season_data,
        x="gw",
        y="total_points",
        color="name",
        use_container_width=True,
        x_label=f"Season: {curr_season}",
        y_label="Points per GW"
    )

    st.subheader("Radar Analysis: Players")
    fig1 = radar_plot_x_stats(
        player_season_data,
        selected_player,
        [s for s in players_x_stats if s in data.columns],
        curr_season
    )
    st.plotly_chart(fig1)

    st.subheader("Radar Analysis: Teams")
    fig2 = radar_plot_x_stats(
        player_season_data,
        selected_player,
        [s for s in teams_x_stats if s in data.columns],
        curr_season
    )
    st.plotly_chart(fig2)

    st.subheader(f"Predictions for the gameweek: {int(preds.gw.max())}")
    preds_data = preds[(preds.gw == preds.gw.max())][["name", "y_pred"]]
    for p in selected_player:
        p_match = preds_data.loc[preds_data['name'] == p, 'y_pred']
        if not p_match.empty:
            val = p_match.item()
            st.metric(label=p, value=round(val, 2))
        else:
            st.write(f"No prediction for {p}")