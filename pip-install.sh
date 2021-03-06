#!/bin/bash

# This is rather annoying. Work around pip's dependency resolver
# by installing each package one at a time. Otherwise, installation takes
# forever as pip downloads every version of the un-pinned dependencies.
# I think this has to do with installing dask/distributed from source.

# Then create the Coiled env with `coiled env create --name batched-send --post-build pip-install.sh --container python:3.9.6-buster`

while read -r line; do
    pip install "$line"
done <<EOF
wheel==0.36.2
flake8==3.9.2
black==21.6b0
coiled==0.0.43
numpy==1.21.0
pandas==1.3.0
jupyterlab==3.0.16
ipython==7.25.0
seaborn==0.11.1
statsmodels==0.12.2
git+https://github.com/dask/dask.git@main
git+https://github.com/gjoseph92/distributed.git@e147a7493c400124e93ef329ec4d874640382fa4
EOF
