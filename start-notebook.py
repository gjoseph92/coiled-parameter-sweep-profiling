#!/usr/bin/env python3

import webbrowser
import coiled

# Because https://gitlab.com/coiled/cloud/-/issues/3173 makes it annoying
# to start a notebook through the UI.

if __name__ == "__main__":
    coiled.start_job("batched-send")
    jobs = sorted(coiled.list_jobs().values(), key=lambda j: j["id"], reverse=True)
    job_id = jobs[0]["id"]
    url = f"https://cloud.coiled.io/job/{job_id}/"
    print(f"Job (probably) at {url}")
    webbrowser.open_new_tab(url)
