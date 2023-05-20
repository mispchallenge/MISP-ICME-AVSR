#!/usr/bin/env bash

addids=
files=
outfile=
set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


addids=(${addids//,/})
files=(${files//,/})
mkdir -p ${outfile%/*}
for i in $(seq 0 `expr ${#addids[@]} - 1`); do
    addid=${addids[i]}
    file=${files[i]}
    sed -e "s/^/${addid}_/" $file
done > $outfile