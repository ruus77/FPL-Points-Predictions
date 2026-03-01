import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .colors_config import team_colors


def best_players_plots(df: pd.DataFrame, num_players: int, season: int) -> None:
    best_df = (df[df.season_id == season]
               .groupby(["web_name", "team_name"])["event_points"].sum()
               .reset_index()
               .sort_values("event_points", ascending=False)
               .head(num_players))

    plt.figure(figsize=(12, 6))
    ax1 = sns.barplot(data=best_df, x="event_points", y="web_name", hue="team_name",
                      palette=team_colors, dodge=False, legend=False)
    for container in ax1.containers:
        ax1.bar_label(container)
    plt.title(f"Top {num_players} zawodników - Suma punktów ({season})")
    plt.show()

    plt.figure(figsize=(12, 6))

    line_data = (df[(df.season_id == season) & (df.web_name.isin(best_df["web_name"]))]
                 .sort_values(["web_name", "gw"])
                 .assign(cum_pts=lambda x: x.groupby("web_name")["event_points"].cumsum()))

    player_to_team = dict(zip(best_df.web_name, best_df.team_name))
    player_palette = {name: team_colors.get(player_to_team[name], "#808080") for name in best_df.web_name}

    sns.lineplot(data=line_data, x="gw", y="cum_pts", hue="web_name",
                 palette=player_palette,
                 marker="o")

    plt.title(f"Skumulowana suma punktów najlepszych piłkarzy ({season})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Zawodnik")
    plt.grid(True, alpha=0.2)
    plt.show()


def expected_stats_vs_actuals(df: pd.DataFrame, teams: pd.DataFrame, season: int,
                              expected_stat: str, actual_stat: str, position: str | list[str]) -> plt.Figure:
    # POPRAWKA: Dodane sharex=True, to absolutnie kluczowe dla poprawnych wniosków
    fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharey=True, sharex=True)
    pos_list = position if isinstance(position, list) else [position]

    for i in range(3):
        # Zakładam, że teams.iloc[i, 0] to liderzy, a teams.iloc[-(i + 1), 0] to doły tabeli
        for j, team in enumerate([teams.iloc[i, 0], teams.iloc[-(i + 1), 0]]):
            plot_data = df[(df.team_name == team) & (df.season_id == season) & (df.position.isin(pos_list))]
            melted_data = plot_data[[expected_stat, actual_stat]].melt()

            sns.barplot(data=melted_data,
                        y="variable",
                        x="value",
                        estimator="sum",
                        errorbar=None,
                        ax=ax[i, j],
                        palette=["#d3d3d3", "#ff7f0e"]  # Szary dla expected, pomarańczowy dla actual
                        )

            ax[i, j].set_title(f"{team}", fontweight='bold')
            ax[i, j].set_ylabel("")

            # Podpisy osi X tylko na samym dole, żeby nie zaśmiecać wykresu
            ax[i, j].set_xlabel("Suma statystyki" if i == 2 else "")

            # POPRAWKA: Wyświetlanie wartości liczbowych na końcach słupków
            for container in ax[i, j].containers:
                ax[i, j].bar_label(container, fmt='%.1f', padding=5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"{expected_stat} vs {actual_stat} dla pozycji {', '.join(pos_list)} (Sezon: {season})", fontsize=16)

    return fig