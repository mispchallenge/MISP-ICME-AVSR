#!/usr/bin/env bash

set -e
. ./cmd.sh
. ./path.sh

pt_dir=/train13/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr
tagdir=/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/org
grain=tri3
stage=0
. ./utils/parse_options.sh
echo "$0 $@"  # Print the command line for logging


  for pos in far near middle;do
    for setclass in train dev; do
      if [[ "train dev" =~ $setclass ]] && [ $pos == far ];then  
        subptdir=$pt_dir/${setclass}_${pos}_gss_${grain}_ali
      else 
        subptdir=$pt_dir/${setclass}_${pos}_${grain}_ali
      fi
      
      if [ -e $subptdir/num_pdf ];then 
        echo "$subptdir/num_pdf"
        pdfcnum=$(cat $subptdir/num_pdf)
      else
        pdfcnum=$pdfcnum
      fi
      pdf_fname=${setclass}_${pos}_pdf_${grain}_${pdfcnum}
        subdir=${tagdir}/pdf_${grain}
        python local/lip_roi.py --pt_dir $subptdir/pt --roiscpdir ${subdir} --filename ${pdf_fname}.scp
        cat ${subdir}/${pdf_fname}.scp | sort -k1 > ${subdir}/${pdf_fname}_tmp.scp
        rm  ${subdir}/${pdf_fname}.scp
        mv ${subdir}/${pdf_fname}_tmp.scp ${subdir}/${pdf_fname}.scp
    done
 done

        



