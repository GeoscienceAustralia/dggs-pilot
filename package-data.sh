#!/bin/bash

zip --password $(cat passwd.txt) \
    -r -9 \
    dggs-sample-data.zip \
    ./data \
    -x '*/.DS_Store' \
    -x '*/README.md'
