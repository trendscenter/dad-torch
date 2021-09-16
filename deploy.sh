#!/bin/bash -e

python setup.py clean sdist
LATEST_RELEASE="dist/$(ls -t1 dist|  head -n 1)"
pip install $LATEST_RELEASE