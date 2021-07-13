#!/usr/bin/env python3

import coiled

if __name__ == "__main__":
    coiled.create_job_configuration(
        name="batched-send",
        software="gjoseph92/batched-send",
        cpu=2,
        memory="4 GiB",
        command=[
            "/bin/bash",
            "-c",
            "SHELL=/bin/bash jupyter lab --allow-root --ip=0.0.0.0 --no-browser",
        ],
        ports=[8888],
        files=["run.py", "util.py"],
    )
