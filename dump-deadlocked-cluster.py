from collections import deque
import json
import sys

import distributed
import coiled


def _normalize(o, simple=False):
    from distributed.scheduler import TaskState as TSScheduler
    from distributed.scheduler import WorkerState
    from distributed.worker import TaskState as TSWorker

    try:
        # Blacklist runspec since this includes serialized functions
        # and arguments which might include sensitive data
        blacklist_attributes = [
            "runspec",
            "_run_spec",
            "exception",
            "traceback",
            "_exception",
            "_traceback",
        ]

        if isinstance(o, dict):
            return {
                _normalize(k, simple=simple): _normalize(v, simple=simple)
                for k, v in o.items()
            }
        elif isinstance(o, set):
            return [_normalize(el, simple=simple) for el in o]
        elif isinstance(o, (deque, tuple, list)):
            return [_normalize(el, simple=simple) for el in o]
        elif isinstance(o, WorkerState):
            res = o.identity()
            res["memory"] = {
                "managed": o.memory.managed,
                "managed_in_memory": o.memory.managed_in_memory,
                "managed_spilled": o.memory.managed_spilled,
                "unmanaged": o.memory.unmanaged,
                "unmanaged_recent": o.memory.unmanaged_recent,
            }
            return res
        elif isinstance(o, TSScheduler):
            if simple:
                # Due to cylcic references in the dependent/dependency graph
                # mapping this causes an infinite recursion
                return str(o)
            base = {
                "type": str(type(o)),
                "repr": str(o),
            }
            base.update(
                {
                    s: _normalize(getattr(o, s), simple=True)
                    for s in TSScheduler.__slots__
                    if s not in blacklist_attributes
                }
            )
            return base
        elif isinstance(o, (TSWorker, TSScheduler)):
            if simple:
                # Due to cylcic references in the dependent/dependency graph
                # mapping this causes an infinite recursion
                return str(o)
            return _normalize(
                {k: v for k, v in o.__dict__.items() if k not in blacklist_attributes},
                simple=True,
            )
        else:
            return str(o)
    except Exception as e:
        raise RuntimeError(repr(o)) from e


def get_worker_info(dask_worker):
    import dask

    return _normalize(
        {
            "status": dask_worker.status,
            "ready": dask_worker.ready,
            "constrained": dask_worker.constrained,
            "long_running": dask_worker.long_running,
            "executing_count": dask_worker.executing_count,
            "in_flight_tasks": dask_worker.in_flight_tasks,
            "in_flight_workers": dask_worker.in_flight_workers,
            "paused": dask_worker.paused,
            "log": dask_worker.log,
            "tasks": dask_worker.tasks,
            "memory_limit": dask_worker.memory_limit,
            "memory_target_fraction": dask_worker.memory_target_fraction,
            "memory_spill_fraction": dask_worker.memory_spill_fraction,
            "memory_pause_fraction": dask_worker.memory_pause_fraction,
            "logs": dask_worker.get_logs(),
            "config": dict(dask.config.config),
        }
    )


def get_scheduler_info(dask_scheduler):
    import dask

    return _normalize(
        {
            "transition_log": dask_scheduler.transition_log,
            "log": dask_scheduler.log,
            "tasks": dask_scheduler.tasks,
            "workers": dask_scheduler.workers,
            "logs": dask_scheduler.get_logs(),
            "config": dict(dask.config.config),
        }
    )


if __name__ == "__main__":
    cluster_name = sys.argv[1]
    print(f"Dumping cluster {cluster_name!r}")

    cluster = coiled.Cluster(name=cluster_name)
    client = distributed.Client(cluster)

    print("Getting scheduler info...")
    scheduler_info = client.run_on_scheduler(get_scheduler_info)
    print("Got scheduler info. Writing...")

    with open(f"{cluster_name}-scheduler.json", "w") as f:
        json.dump(scheduler_info, f)
        print(f"Wrote scheduler state to {f.name}")
    del scheduler_info

    print("Getting worker info...")
    worker_info = client.run(get_worker_info)
    print("Got worker info. Writing...")
    with open(f"{cluster_name}-workers.json", "w") as f:
        json.dump(worker_info, f)
        print(f"Wrote worker state to {f.name}")
    del worker_info
