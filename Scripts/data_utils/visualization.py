# data_utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import team_colors


def best_players_plots(df: pd.DataFrame, num_players: int, season: str = "2024-25"):
    best_df = (df[df.season_id == season]
               .groupby(["name", "team"])["total_points"].sum()
               .reset_index()
               .sort_values("total_points", ascending=False)
               .head(num_players))

    plt.figure(figsize=(12, 5))
    ax1 = sns.barplot(data=best_df, x="total_points", y="name", hue="team",
                      palette=team_colors, dodge=False, legend=False)
    for container in ax1.containers:
        ax1.bar_label(container)
    plt.title(f"Top {num_players} zawodników - Suma punktów ({season})")
    plt.show()

    plt.figure(figsize=(12, 6))
    line_data = (df[(df.season_id == season) & (df.name.isin(best_df["name"]))]
                 .sort_values(["name", "gw"])
                 .assign(cum_pts=lambda x: x.groupby("name")["total_points"].cumsum()))

    sns.lineplot(data=line_data, x="gw", y="cum_pts", hue="name",
                 palette={n: team_colors.get(t) for n, t in zip(best_df.name, best_df.team)},
                 marker="o")
    plt.title(f"Skumulowana suma punktów najlepszych piłkarzy ({season})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.show()


def expected_stats_vs_actuals(df: pd.DataFrame, teams: pd.DataFrame, season: str,
                              expected_stat: str, actual_stat: str, position: any) -> plt.Figure:
    fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharey=True)
    pos_list = position if isinstance(position, list) else [position]

    for i in range(3):
        for j, team in enumerate([teams.iloc[i, 0], teams.iloc[-(i + 1), 0]]):
            plot_data = df[(df.team == team) & (df.season_id == season) & (df.position.isin(pos_list))]
            sns.barplot(data=plot_data[[expected_stat, actual_stat]].melt(),
                        y="variable", x="value", estimator="sum", errorbar=None, ax=ax[i, j])
            ax[i, j].set_title(f"{team}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"{expected_stat} vs {actual_stat} dla pozycji {', '.join(pos_list)}")
    return fig