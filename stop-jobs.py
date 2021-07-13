#!/usr/bin/env python3

import coiled

if __name__ == "__main__":
    for name, job in coiled.list_jobs().items():
        print(f"Stopping {job['id']} ({name!r})")
        coiled.stop_job(name)
