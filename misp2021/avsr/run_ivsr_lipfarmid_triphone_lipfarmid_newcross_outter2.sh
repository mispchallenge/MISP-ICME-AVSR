#!/usr/bin/env bash
source ./bashrc
set -eou pipefail

train_set=av/train_far_farmidlip 
valid_set=av/dev_far_farlip
test_sets=av/sum_far_farlip
asr_config=conf/train_avsr_outter_2vblock.yaml
avsr_exp=exp_far/train_avsr_outter_2vblock 
gpu_command= #CUAD_VISIBLE_DIVICES=0
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml
use_lm=true
use_word_lm=false
opts=""
if [[ -n $gpu_command ]];then 
    opts+="--gpu_command $gpu_command"
fi


./avsr.sh                                  \
    --stage 0                              \
    --stop_stage 0                         \
    --avsr_exp ${avsr_exp}                 \
    --lang zh                              \
    --nj 8                                 \
    --speed_perturb_factors "0.9 1.0 1.1"  \
    --inference_asr_model  valid.acc.ave_5best.pth \
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 1                               \
    --token_type char                      \
    --feats_type raw                       \
    --use_lm ${use_lm}                     \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --feats_normalize "utterance_mvn"      \
      $opts                                \
    --lm_train_text "data/${train_set}/text" "$@" 
    
    
