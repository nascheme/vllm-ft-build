#!/bin/bash

# Make folder for models cache.  Note that cache/token likely needs to
# exist in order to download some models.
if ! test -d; then
    mkdir cache
fi

export HF_HOME=`pwd`/cache
export PYTHON_GIL=0

cd test
uv run ./simple_generate.py
