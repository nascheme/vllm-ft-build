#!/bin/bash

# Make folder for models cache.  Note that cache/token likely needs to
# exist in order to download some models.
if ! test -d; then
    mkdir cache
fi

export HF_HOME=`pwd`/cache

case $(uv run python -c 'import sys; print(sys.flags.gil)') in
    None)
        # the GIL-enabled build gives an error if this is set
        export PYTHON_GIL=0
        ;;
esac

cd test
uv run ./simple_generate.py
