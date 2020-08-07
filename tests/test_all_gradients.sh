#!/bin/bash

#usage: `bash tests/test_all_gradients.sh` (works when called from anywhere)

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "testing all gradient checkers in $TEST_DIR"

# runs all of the gradient specific tests
python $TEST_DIR/test_gradients_fdfd.py
python $TEST_DIR/test_gradients_fdfd_mf.py
python $TEST_DIR/test_gradients_fdtd.py
python $TEST_DIR/test_primitives.py