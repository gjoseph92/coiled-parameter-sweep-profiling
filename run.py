from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional, Union
import asyncio
import itertools
import csv
from pathlib import Path
import random
import time
import traceback
import sys
import signal

import coiled
import distributed
import dask
import dask.array as da
import pandas as pd
from rich import print
from rich.progress import Progress

from util import gather_with_concurrency, graph_size

Cluster = Union[distributed.deploy.Cluster, coiled.Cluster[coiled.cluster.Async]]


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
    await distributed.client._wait(client.persist(shuffled))
    elapsed = time.perf_counter() - start
    return graph_size(shuffled), elapsed


async def blockwise(client: distributed.Client) -> tuple[int, float]:
    "Embarrasingly-parallel workload with no communication and simple scheduling"
    n_workers = len(await client.nthreads())

    # Target: 7k tasks per worker
    arr = da.random.random(n_workers * 1000, chunks=1)
    for _ in range(6):
        arr = arr + 1

    start = time.perf_counter()
    # TODO distributed.wait doesn't work with multiple clients
    await distributed.client._wait(client.persist(arr, optimize_graph=False))
    # ^ Use unoptimized graph to prevent blockwise fusion, so there are many embarrasingly parallel tasks
    elapsed = time.perf_counter() - start
    return graph_size(arr, optimize_graph=False), elapsed


WORKLOADS: dict[str, Callable[[distributed.Client], Awaitable[tuple[int, float]]]] = {
    "shuffle": shuffle,
    "blockwise": blockwise,
}


async def make_cluster_coiled(batched_send_interval: str, compression: str) -> Cluster:
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
            # Lengthen timeouts to reduce errors
            "DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP": "60s",
            "DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT": "60s",
        },
    )


async def make_cluster_local(batched_send_interval: str, compression: str) -> Cluster:
    # TODO set `batched_send_interval` and `compression` locally.
    # This is just for testing things don't explode.
    return await distributed.LocalCluster(
        n_workers=1,
        processes=True,
        threads_per_worker=1,
        asynchronous=True,
    )


async def dump_cluster_state(cluster: Cluster, timeout=10 * 60) -> bool:
    """
    Run the `dump-deadlocked-cluster.py` script as a subprocess.

    We use a subprocess to avoid blocking the event loop here transferring lots of data,
    or any other bad behavior that might occur talking to this cluster.
    """
    assert cluster.name is not None
    dumps = Path("dumps")
    dumps.mkdir(exist_ok=True)
    with open(dumps / f"{cluster.name}-dumplog.txt", "w") as f:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            Path("dump-deadlocked-cluster.py").resolve(),
            cluster.name,
            stdout=f,
            stderr=f,
            cwd=dumps,
        )
        start = time.perf_counter()
        try:
            retcode = await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            print(
                f"[red]Timed out dumping cluster state for {cluster.name}. See {f.name} for logs."
            )
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            return False

        elapsed = time.perf_counter() - start
        if retcode == 0:
            print(
                f"[green]Successfully dumped cluster state for {cluster.name} in {elapsed:.1f}sec!"
            )
            return True
        print(
            f"[red]Failed dumping cluster state for {cluster.name} in {elapsed:.1f}sec. See {f.name} for logs."
        )
        return False


async def trial(
    cluster_size: int,
    batched_send_interval: str,
    workload: str,
    compression: str,
    repetition: int,
) -> tuple[int, float, dict[str, list[tuple[int, int]]]]:
    client: Optional[distributed.Client] = None
    cluster: Optional[Cluster] = None
    workload_func = WORKLOADS[workload]
    try:
        print(
            f"[italic]Starting cluster for "
            f"{cluster_size=} {batched_send_interval=} {workload=} {compression=} {repetition=}"
        )
        cluster = await make_cluster_coiled(batched_send_interval, compression)
        client = await distributed.Client(
            cluster, asynchronous=True, set_as_default=False
        )
        assert client is not None
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

        DEADLOCK_TIMEOUT = 15 * 60
        try:
            tasks, runtime = await asyncio.wait_for(
                workload_func(client), DEADLOCK_TIMEOUT
            )
        except asyncio.TimeoutError:
            print(
                f"[bold red]Suspected deadlock on {cluster.name}. Dumping state...[/] "
                f"{cluster_size=} {batched_send_interval=} {workload=} {compression=} {repetition=}"
            )
            dumped = await dump_cluster_state(cluster, timeout=15 * 60)
            raise RuntimeError(
                f"Deadlocked cluster {cluster.name} after {DEADLOCK_TIMEOUT} sec. Successful dump: {dumped}"
            ) from None

        try:
            batched_send_stats = await client.run(
                lambda dask_worker: (dask_worker.batched_stream.buffer_sizes_at_send)
            )
        except Exception as e:
            print(
                f"[bold red]Error getting batched send stats: {e} "
                f"{cluster_size=} {batched_send_interval=} {workload=} {compression=} {repetition=}"
            )
            traceback.print_exc()
            batched_send_stats = {}
        assert batched_send_stats is not None
        return tasks, runtime, batched_send_stats

    finally:
        if client:
            await client.shutdown()
            await client.close()
        if cluster:
            await cluster.close()


async def run_trial(
    writer: csv.DictWriter, writer_batched_send: csv.DictWriter, vars: dict
) -> None:
    try:
        tasks, runtime, batched_send_stats = await trial(**vars)
    except Exception as e:
        print(f"[bold red]Error during {vars}: {e}")
        traceback.print_exc()
        tasks = runtime = float("nan")
        batched_send_stats = {}
        error = repr(e)
    else:
        error = None
        print(
            f"[bold blue]{vars}: {runtime} sec, {tasks} tasks, "
            f"{sum(map(len, batched_send_stats.values()))} batched sends"
        )

    writer.writerow({**vars, "runtime": runtime, "tasks": tasks, "error": error})

    for worker, (sends) in batched_send_stats.items():
        for i, ((buffer_len, nbytes)) in enumerate(sends):
            writer_batched_send.writerow(
                {
                    **vars,
                    "worker": worker,
                    "i": i,
                    "buffer_len": buffer_len,
                    "nbytes": nbytes,
                }
            )


async def run(
    *,
    csv_path: str,
    batched_send_csv_path: str,
    coiled_parallelism: int,
    variables: dict[str, List[Any]],
) -> None:
    keys = list(variables)
    combos = [dict(zip(keys, vs)) for vs in itertools.product(*variables.values())]
    random.shuffle(combos)

    with open(csv_path, "w", newline="") as f, open(
        batched_send_csv_path, "w", newline=""
    ) as f_batched_send:
        writer = csv.DictWriter(f, keys + ["runtime", "tasks", "error"])
        writer.writeheader()

        writer_batched_send = csv.DictWriter(
            f_batched_send, keys + ["worker", "i", "buffer_len", "nbytes"]
        )
        writer_batched_send.writeheader()

        with Progress() as progress:
            task = progress.add_task("Running trials", total=len(combos))
            global print
            old_print = print
            print = progress.console.print

            async def _run_trial(vars: dict):
                await run_trial(writer, writer_batched_send, vars)
                progress.advance(task)
                f.flush()
                f_batched_send.flush()

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
            "distributed.comm.timeouts.connect": "60s",
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
        workload=[
            "shuffle",
            # "blockwise",
        ],
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
            batched_send_csv_path="batched-sends.csv",
            coiled_parallelism=4,
            variables=variables,
        )
    )
