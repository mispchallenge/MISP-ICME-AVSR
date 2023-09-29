#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=0
need_channel=
self_fix=
pos=far
stop_stage=3
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


feature_dir= # the path to store gss audio 
dataset_dir= #the path to orginal mutiple waveform files

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare misp2021 datadir in kaldi format for trainset,devset,evalset and addtionset"
  mkdir -p data
  for pos in far;do
    for setclass in train ; do
      [[ {"middle","near"} =~ $pos ]] && [[ {"dev","eval","add"} =~ $setclass ]] && continue
      datadir=${self_fix}gss_${setclass}_${pos}
      wavdir=${feature_dir}/${setclass}_wave/${pos}/wpe/gss/wav
      [[ $setclass == "add" ]] && setclass="addition"
      mp4dir=${dataset_dir}/${setclass}_${pos}_video/mp4
      textdir=${dataset_dir}/${setclass}_near_transcription/TextGrid
      [[ $pos == "near" ]] && [[ $setclass == "train" ]] && datadir=${self_fix}${setclass}_${pos} && wavdir=${feature_dir}/${setclass}_wave/near/wav
      
      if [[ ! -f data/$datadir/.done ]]; then
        # feature_dir offer augmentation wav, datadir offer video an textgrid , dataset_dir offer original mutiple channel waveform
        local/prepare_gss_data.sh  ${wavdir} ${mp4dir} ${textdir} $setclass data/${datadir} || exit 1;
        touch data/${datadir}/.done
      fi
    done
  done
fi


#comebine eval and addition
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: comebine eval and addition"
  pos=far
  data_dir1=data/${self_fix}gss_eval_${pos}
  data_dir2=data/${self_fix}gss_add_${pos}
  utils/combine_data.sh data/${self_fix}gss_sum_${pos} $data_dir1 $data_dir2
fi
