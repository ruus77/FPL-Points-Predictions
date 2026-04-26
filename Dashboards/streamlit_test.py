import pandas as pd
import streamlit as st
import sys
from pathlib import Path
import importlib.util

root_path = Path(__file__).resolve().parent.parent

if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

config_file_path = root_path / "config.py"
spec = importlib.util.spec_from_file_location("config", config_file_path)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

from data_utils.visualization import radar_plot_x_stats
from data_utils.colors_config import players_x_stats, teams_x_stats

@st.cache_data
def load_data():
    df = pd.read_parquet(config.TIDY_DATA_PATH)
    df_ema = pd.read_parquet(config.FPL_DATA_PATH)
    df_preds = pd.read_parquet(config.PREDICTED_STATS_PATH)

    for _df in [df, df_ema, df_preds]:
        _df.columns = _df.columns.str.strip().str.lower()
    
    return df, df_ema, df_preds

data, data_ema, preds = load_data()

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

    player_season_data_ema = data_ema[(data_ema.name.isin(selected_player)) & (data_ema.season == curr_season)]
    player_season_data_ema = player_season_data_ema.sort_values("gw")

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
    st.subheader("Transfer Balance History")
    st.line_chart(
        data=player_season_data,
        x="gw",
        y="transfers_balance",
        color="name",
        use_container_width=True,
        x_label=f"Season: {curr_season}",
        y_label="Transfer Balance"
    )
    st.subheader("Radar Analysis: Players")
    fig1 = radar_plot_x_stats(
        player_season_data,
        selected_player,
        [s for s in players_x_stats if s.lower() in data.columns],
        [curr_season]
    )
    st.plotly_chart(fig1)

    st.subheader("Radar Analysis: Teams")
    fig2 = radar_plot_x_stats(
        player_season_data,
        selected_player,
        [s for s in teams_x_stats if s.lower() in data.columns],
        [curr_season]
    )
    st.plotly_chart(fig2)

    st.subheader("Predicted points")
    cols = st.columns(len(selected_player))

    for i, p in enumerate(selected_player):
        p_preds = preds[preds.name == p]

        if not p_preds.empty:
            latest_prediction = p_preds.iloc[-1]["y_pred"]

            with cols[i]:
                st.metric(
                    label=f"{p}",
                    value=f"{latest_prediction:.2f}"
                )
        else:
            with cols[i]:
                st.info(f"{p}: No preds")

else:
    st.info("Please select at least one player from the sidebar to see the analysis.")
