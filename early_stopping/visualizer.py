import json

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from .simulator import StoppingSimulator


class Visualizations:
    def __init__(self, simulator: StoppingSimulator):
        self.simulator = simulator

    def plot_strategy(self, strategy: str, return_fig=False) -> matplotlib.figure:
        eval_sets, params = self._preprocess_ranks(strategy)

        nparam = len(params)
        if nparam == 1:
            plot_func = self.plot_1d
        elif nparam == 2:
            plot_func = self.plot_2d
        elif nparam == 3:
            plot_func = self.plot_3d
        else:
            raise ValueError("Plots with this many dimensions are not supported")

        fig = plot_func(strategy, eval_sets, *params)

        if return_fig:
            return fig

    def plot_1d(self, strategy: str, eval_sets: dict, param: str) -> matplotlib.figure:
        fig, ax = plt.subplots()

        for eval_set, ranks in eval_sets.items():
            ranks[param] = ranks["params"].apply(lambda x: x[param])
            sns.lineplot(x=param, y="rank", data=ranks, ax=ax, label=eval_set)

        plt.legend()

        plt.title(f"Stopping Strategy Performance for {strategy}")
        plt.xlabel(param)
        plt.ylabel("rank")
        plt.grid(True)
        plt.show()

        return fig

    def plot_2d(
        self, strategy: str, eval_sets: dict, x_param: str, y_param: str
    ) -> matplotlib.figure:
        fig, axes = plt.subplots(
            1,
            len(eval_sets),
            figsize=(5 * len(eval_sets), 4),
            sharex=True,
            sharey=True,
            squeeze=0,
        )
        fig.suptitle(f"Stopping Strategy Performance for {strategy}")
        axes = axes.flatten()

        for ax, (eval_set, ranks) in zip(axes, eval_sets.items()):
            xy = ranks["params"].apply(lambda r: (r[x_param], r[y_param])).tolist()
            z = ranks["rank"]

            x_values = sorted(set(c[0] for c in xy))
            y_values = sorted(set(c[1] for c in xy))

            heatmap = np.zeros((len(y_values), len(x_values)))

            for (x, y), rank in zip(xy, z):
                heatmap[y_values.index(y), x_values.index(x)] = rank

            ax.set_title(f"{eval_set}")
            plot = ax.imshow(
                heatmap,
                cmap="viridis",
                origin="lower",
                aspect="auto",
                extent=(min(x_values), max(x_values), min(y_values), max(y_values)),
            )

        fig.subplots_adjust(top=0.85)
        fig.colorbar(plot, ax=axes, label="rank")

        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.show()

        return fig

    def plot_3d(
        self, strategy: str, eval_sets: dict, x_param: str, y_param: str, z_param: str
    ) -> matplotlib.figure:
        # FIXME: contour lines are not especially useful for visualizing effects of z_param
        fig, axes = plt.subplots(
            1,
            len(eval_sets),
            figsize=(5 * len(eval_sets), 4),
            sharex=True,
            sharey=True,
            squeeze=0,
        )
        fig.suptitle(f"Stopping Strategy Performance for {strategy}")
        axes = axes.flatten()

        for ax, (eval_set, ranks) in zip(axes, eval_sets.items()):
            xy = ranks["params"].apply(lambda r: (r[x_param], r[y_param])).tolist()
            z = ranks["params"].apply(lambda r: r[z_param]).tolist()
            ranks = ranks["rank"]

            x_values = sorted(set(c[0] for c in xy))
            y_values = sorted(set(c[1] for c in xy))

            heatmap = np.zeros((len(y_values), len(x_values)))

            for (x, y), rank in zip(xy, ranks):
                heatmap[y_values.index(y), x_values.index(x)] = rank

            ax.set_title(f"{eval_set}")
            plot = ax.imshow(
                heatmap,
                cmap="Blues",
                origin="lower",
                aspect="auto",
                extent=(min(x_values), max(x_values), min(y_values), max(y_values)),
            )

            # Create grid for contour plot
            grid_x, grid_y = np.meshgrid(x_values, y_values)
            grid_z = griddata(xy, z, (grid_x, grid_y), method="linear")
            contour = ax.contour(
                grid_x,
                grid_y,
                grid_z,
                levels=6,
                colors="black",
                linewidths=0.8,
                alpha=0.8,
            )
            ax.clabel(contour, inline=1, fontsize=10)

        fig.subplots_adjust(top=0.85)
        fig.colorbar(plot, ax=axes, label="rank")

        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.show()

        return fig

    def _preprocess_ranks(self, strategy: str) -> tuple[dict, set]:
        params = set()
        eval_sets = {}
        for eval_set, ranks in self.simulator.ranks.items():
            if strategy not in ranks["strategy"].values:
                raise ValueError(
                    f"Simulator does not have results for strategy: {strategy}"
                )

            df = ranks.copy()
            df = df[df["strategy"] == strategy]

            paramdf = pd.DataFrame(df["params"].tolist())
            if len(paramdf.columns) != len(set(paramdf.columns)):
                raise ValueError("Strategy simulations have conflicting parameters")
            elif not params:
                params = set(paramdf.columns)
            elif set(paramdf.columns) != params:
                raise ValueError("Strategy simulations have conflicting parameters")

            eval_sets[eval_set] = df

        params = list(params)
        params.sort()
        return eval_sets, params
