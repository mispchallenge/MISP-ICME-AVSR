#!/usr/bin/env bash

set -e
. ./cmd.sh
. ./path.sh
pt_dir=avsrmisp2021_avsr/feature/misp2021_avsr
tagdir=avsr/dump/raw/org/rois
. ./utils/parse_options.sh



echo "$0 $@"  # Print the command line for logging



  for pos in far middle;do
    for setclass in train dev eval addition ; do
      python local/lip_roi.py --pt_dir $pt_dir/${setclass}_${pos}_video_lip_segment/pt --roiscpdir $tagdir --filename ${setclass}_${pos}_roi.scp
      cat $tagdir/${setclass}_${pos}_roi.scp | sort -k1 > $tagdir/${setclass}_${pos}_roi_tmp.scp
      rm  $tagdir/${setclass}_${pos}_roi.scp
      mv $tagdir/${setclass}_${pos}_roi_tmp.scp $tagdir/${setclass}_${pos}_roi.scp
    done
 done

for pos in far middle;do
  for setclass in eval addition ;do
    cat $tagdir/${setclass}_${pos}_roi.scp 
  done | sort -k1 > $tagdir/sum_${pos}_roi.scp 
done

for pos in far middle;do
  for setclass in eval addition ;do
    rm $tagdir/${setclass}_${pos}_roi.scp 
  done 
done
