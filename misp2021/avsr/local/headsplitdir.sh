#!/usr/bin/env bash

# transform misp data to kaldi format
# this script use to cut a big kaldi data into a small one  save_num can control its size

set -e -o pipefail
echo "$0 $@"
save_num=100
inputdir=dump/raw/av_pdf/av_farpdf/dev_far_farmidlip
outputdir=dump/raw/av_pdf/av_farpdf/dev_far_farmidlip_100
. ./cmd.sh || exit 1
. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1


mkdir -p $outputdir
[[ ! -e ${inputdir}/wav.scp ]] && echo "there is no wav.scp in ${inputdir}" && exit 1
for file in pdf_fps feats_type;do 
    if [ -f ${inputdir}/$file ];then 
    cat ${inputdir}/$file > ${outputdir}/$file
    fi
done

if [ -e  ${inputdir}/spk2utt ];then 
    cat ${inputdir}/spk2utt > ${outputdir}/spk2utt
fi

for file in wav.scp utt2spk text segments text_shape.char speech_shape reco2dur;do
    if [[ -e $inputdir/$file ]] ;then 
       head -n $save_num $inputdir/$file > ${outputdir}/$file
    fi 
done

utils/fix_data_dir.sh  ${outputdir}

