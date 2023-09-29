#!/usr/bin/env bash
# extract region of interest (roi) in the video, store as npz file, item name is "data"

set -e
# configs for 'chain'
drop_threshold=
nj=15
python_path=
roi_type=head
roi_size="96 96"
need_speaker=true 
roi_sum=
local=true
poscustome_fix=false
cuda_index=0
snapshot_path=
# End configuration section.
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $# != 3 ]; then
  echo "Usage: $0 <data-set> <roi-json-dir> <roi-store-dir>"
  echo " $0 data/train_far /path/roi data/train_far_sp_hires"
  exit 1;
fi
echo "$0 $@"  # Print the command line for logging

data_dir=$1
roi_json_dir=$2
roi_store_dir=$3

optional="--roi_type $roi_type --roi_size $roi_size"
if $need_speaker; then
  optional="$optional --need_speaker"
fi
if $roi_sum; then
  optional="$optional --roi_sum"
fi
if [ -n $drop_threshold ];then 
  optional="$optional --drop_threshold $drop_threshold"
fi 

optional="$optional --static_dir $roi_store_dir/static"

if $poscustome_fix ;then 
  optional="$optional --poscustome_fix"
fi

mkdir -p $roi_store_dir
mkdir -p $roi_store_dir/log
###########################################################################
# segment mp4 and crop roi, store as pt
###########################################################################

for n in `seq $nj`; do
  cat <<-EOF > $roi_store_dir/log/roi.$n.sh
  source bashrc
  . ./cmd.sh
  . ./path.sh
  CUDA_VISIBLE_DEVICES=$cuda_index $python_path local/segment_video_roi.py $optional -ji $((n-1)) -nj $nj $data_dir $roi_json_dir $roi_store_dir/pt
EOF
  # $roi_store_dir/samples
done

chmod a+x $roi_store_dir/log/roi.*.sh

$train_cmd JOB=1:$nj $roi_store_dir/log/roi.JOB.log $roi_store_dir/log/roi.JOB.sh || exit 1;
