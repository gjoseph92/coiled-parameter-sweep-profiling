"""
python plots.py <input.csv> <output.png>
"""

import sys

import pandas as pd
import seaborn as sns


def plot_times(df: pd.DataFrame) -> sns.FacetGrid:
    df = df.assign(tasks_per_sec=lambda df: df.tasks / df.runtime)
    return sns.relplot(
        data=df,
        x="batched_send_interval_ms",
        y="tasks_per_sec",
        col="cluster_size",
        # height=3,
    )


def plot_batched_sends(df: pd.DataFrame) -> sns.FacetGrid:
    g = sns.FacetGrid(
        df,
        row="cluster_size",
        col="batched_send_interval_ms",
        margin_titles=True,
        sharey=False,
        sharex=False,
    )
    g.map(sns.histplot, "buffer_len", kde=True, log_scale=(False, False))
    return g


if __name__ == "__main__":
    csv, out_fig = sys.argv[1], sys.argv[2]
    df = pd.read_csv(csv).assign(
        batched_send_interval_ms=(
            lambda df: df.batched_send_interval.str[:-2].astype(int)
        )
    )

    if "buffer_len" in df.columns:
        g = plot_batched_sends(df)
    else:
        g = plot_times(df)

    g.tight_layout()
    g.savefig(out_fig)
