from __future__ import annotations

from typing import TypeVar, Awaitable
import asyncio
import dask

T = TypeVar("T")


async def gather_with_concurrency(n: int, *tasks: Awaitable[T]) -> tuple[T, ...]:
    # https://stackoverflow.com/a/61478547
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


def graph_size(x) -> int:
    return len(dask.base.collections_to_dsk([x], optimize_graph=True))
