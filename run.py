from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional
import asyncio
import itertools
import csv
import random
import time
import traceback

import coiled
import distributed
import dask
import pandas as pd
from rich import print
from rich.progress import Progress

from util import gather_with_concurrency, graph_size


async def shuffle(client: distributed.Client) -> tuple[int, float]:
    n_workers = len(await client.nthreads())

    # target: 100 partitions per worker
    start = pd.to_datetime("2000-01-01")
    partition_freq = pd.to_timedelta("1h")
    end = start + partition_freq * (n_workers * 100)

    df = dask.datasets.timeseries(
        start=start,
        end=end,
        partition_freq=partition_freq,
        freq="1800s",
    )
    shuffled = df.shuffle("id", shuffle="tasks")

    start = time.perf_counter()
    # TODO distributed.wait doesn't work with multiple clients
    await distributed.client._wait(shuffled.persist())
    elapsed = time.perf_counter() - start
    return graph_size(shuffled), elapsed


WORKLOADS: dict[str, Callable[[distributed.Client], Awaitable[tuple[int, float]]]] = {
    "shuffle": shuffle,
}


async def make_cluster_coiled(
    batched_send_interval: str, compression: str
) -> distributed.deploy.Cluster:
    return await coiled.Cluster(
        asynchronous=True,
        n_workers=1,
        software="gjoseph92/batched-send",
        worker_cpu=1,
        worker_memory="4 GiB",
        scheduler_cpu=1,
        scheduler_memory="8 GiB",
        shutdown_on_close=True,
        scheduler_options={"idle_timeout": "1 hour"},
        environ={
            "MALLOC_TRIM_THRESHOLD_": "0",
            "DASK_DISTRIBUTED__WORKER__BATCHED_SEND_INTERVAL": batched_send_interval,
            # NOTE: client <--> scheduler comms are uncompressed (probably) because we can't set
            # the local dask config in an async-safe way, so I believe
            # https://github.com/dask/distributed/blob/e0593fa/distributed/comm/core.py#L150-L153
            # means we will at least _send_ data uncompressed.
            "DASK_DISTRIBUTED__COMM__COMPRESSION": compression,
        },
    )


async def make_cluster_local(
    batched_send_interval: str, compression: str
) -> distributed.deploy.Cluster:
    # TODO set `batched_send_interval` and `compression` locally.
    # This is just for testing things don't explode.
    return await distributed.LocalCluster(
        n_workers=1,
        processes=True,
        threads_per_worker=1,
        asynchronous=True,
    )


async def trial(
    cluster_size: int,
    batched_send_interval: str,
    workload: str,
    compression: str,
    repetition: int,
) -> tuple[int, float]:
    client: Optional[distributed.Client] = None
    cluster: Optional[distributed.deploy.Cluster] = None
    workload_func = WORKLOADS[workload]
    try:
        print(
            f"[italic]Starting cluster for "
            f"{cluster_size=} {batched_send_interval=} {workload=} {compression=} {repetition=}"
        )
        cluster = await make_cluster_local(batched_send_interval, compression)
        client = await distributed.Client(
            cluster, asynchronous=True, set_as_default=False
        )
        try:
            print(
                f"[italic]Scaling for "
                f"{cluster_size=} {batched_send_interval=} {workload=} {compression=} {repetition=}"
            )
            await cluster.scale(cluster_size)
        except asyncio.TimeoutError:
            pass

        await client.wait_for_workers(cluster_size)
        print(
            f"[italic][underline]Here we go[/] - "
            f"{cluster_size=} {batched_send_interval=} {workload=} {compression=} {repetition=}"
        )

        return await workload_func(client)

    finally:
        if client:
            await client.shutdown()
            await client.close()
        if cluster:
            await cluster.close()


async def run_trial(writer: csv.DictWriter, vars: dict) -> None:
    try:
        tasks, runtime = await trial(**vars)
    except Exception as e:
        print(f"[bold red]Error during {vars}: {e}")
        traceback.print_exc()
        tasks = runtime = float("nan")
    else:
        print(f"[bold blue]{vars}: {runtime} sec, {tasks} tasks")

    writer.writerow({**vars, "runtime": runtime, "tasks": tasks})


async def run(
    *, csv_path: str, coiled_parallelism: int, variables: dict[str, List[Any]]
) -> None:
    keys = list(variables)
    combos = [dict(zip(keys, vs)) for vs in itertools.product(*variables.values())]
    random.shuffle(combos)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, keys + ["runtime", "tasks"])
        writer.writeheader()

        with Progress() as progress:
            task = progress.add_task("Running trials", total=len(combos))
            global print
            old_print = print
            print = progress.console.print

            async def _run_trial(vars: dict):
                await run_trial(writer, vars)
                progress.advance(task)
                f.flush()

            await gather_with_concurrency(
                coiled_parallelism, *[_run_trial(vars) for vars in combos]
            )
            print = old_print


if __name__ == "__main__":
    dask.config.set(
        {
            # This is key---otherwise we're uploading ~300MiB of graph to the scheduler
            "optimization.fuse.active": False,
            # Handle flaky connections to Coiled
            "distributed.comm.retry.count": 5,
            # Give clusters a long time to connect?
            "distributed.comm.timeouts.tcp": "60s",
        }
    )

    variables = dict(
        repetition=list(range(5)),
        cluster_size=[
            # 2,
            10,
            20,
            50,
            100,
            # 200,
            # 300,
        ],
        batched_send_interval=[
            "2ms",
            "5ms",
            "10ms",
            "25ms",
            "50ms",
            "75ms",
            "100ms",
            "200ms",
            "500ms",
        ],
        workload=["shuffle"],
        compression=[
            "None",
            # "zlib",
            # "snappy",
            # "lz4",
            # "zstandard",
            # "blosc",
        ],
    )

    asyncio.run(
        run(
            csv_path="results.csv",
            coiled_parallelism=4,
            variables=variables,
        )
    )
