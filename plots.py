"""
python plots.py <input.csv> <output.png>
"""

import sys

import dask.dataframe as dd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_times(df: dd.DataFrame) -> sns.FacetGrid:
    # TODO https://github.com/dask/dask/issues/7923 also it's just faster
    df = df.compute()
    return sns.lmplot(
        data=df.query("batched_send_interval_ms < 500"),
        x="batched_send_interval_ms",
        y="runtime",
        col="cluster_size",
        # robust=True,
        # x_estimator=np.mean,
        x_jitter=2,
        sharey=False,
    )


def plot_times_per_worker(df: dd.DataFrame) -> sns.FacetGrid:
    df = df.assign(
        tasks_per_sec=lambda df: df.tasks / df.runtime,
        tasks_per_sec_per_worker=lambda df: df.tasks / df.runtime / df.cluster_size,
    )
    # TODO https://github.com/dask/dask/issues/7923 also it's just faster
    df = df.compute()
    return sns.lmplot(
        data=df.query("batched_send_interval_ms < 500"),
        x="batched_send_interval_ms",
        y="tasks_per_sec_per_worker",
        hue="cluster_size",
        x_jitter=2,
    )


def plot_times_improvement(df: dd.DataFrame) -> sns.FacetGrid:
    # TODO too much noise for this to be meaningful?
    # The y-intercepts for the trendlines are ~±20% improvement.
    # Feels like it should go through the origin.
    avg_runtimes = (
        df.groupby(["batched_send_interval_ms", "cluster_size"])
        .runtime.mean()
        .compute()
    )
    initial = avg_runtimes.loc[2]  # TODO get the min non-NA value for each cluster_size
    improvement = ((avg_runtimes - initial) / initial) * -100
    return sns.lmplot(
        data=improvement.reset_index(name="percent_improvement").query(
            "batched_send_interval_ms < 500"
        ),
        x="batched_send_interval_ms",
        y="percent_improvement",
        col="cluster_size",
        sharey=False,
    )


def plot_batched_sends(df: dd.DataFrame, nbytes=False) -> sns.FacetGrid:
    # TODO https://github.com/dask/dask/issues/7923 also it's just faster
    df = df.compute()
    g = sns.FacetGrid(
        df,
        row="cluster_size",
        col="batched_send_interval_ms",
        margin_titles=True,
        # sharey=False,
        sharex=False,
        ylim=(0, 1),
    )
    g.map(
        sns.histplot,
        "nbytes" if nbytes else "buffer_len",
        bins=20,
        kde=True,
        log_scale=(True, False),
        stat="probability",
    )
    return g


if __name__ == "__main__":
    sns.set_theme()
    _, *csvs, out_fig = sys.argv

    flag = csvs[0] if csvs[0].startswith("--") else None
    if flag:
        csvs = csvs[1:]

    df = dd.read_csv(csvs).assign(
        batched_send_interval_ms=(
            lambda df: df.batched_send_interval.str[:-2].astype(int)
        )
    )

    if "buffer_len" in df.columns:
        g = plot_batched_sends(df, nbytes=flag == "--nbytes")
    elif flag == "--improvement":
        g = plot_times_improvement(df)
    elif flag == "--per-worker":
        g = plot_times_per_worker(df)
    else:
        g = plot_times(df)

    g.tight_layout()
    g.savefig(out_fig)
