#!/usr/bin/env bash
ref=
hyp=
source ./bashrc
set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
cat $hyp | compute-wer --text --mode=present ark:$ref  ark,p:- | grep WER  || exit 1;
