#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=gss_train_far_sp_cat2 
valid_set=gss_sum_far
test_sets=gss_sum_far
asr_config=conf/train_asr.yaml
asr_exp=exp_far/a_com_sp_cat2 
gpu_command=
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml
use_lm=true
use_word_lm=false
opts=""
if [[ -n $gpu_command ]];then 
    opts+="--gpu_command $gpu_command"
fi
./asr.sh                                   \
    --stage 0                              \
    --stop_stage 0                         \
    --asr_exp ${asr_exp}                   \
    --lang zh                              \
    --nj 8                                 \
    --inference_asr_model  valid.acc.ave_5best.pth \
    --speed_perturb_factors "0.9 1.0 1.1"  \
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
    $opts                                  \
    --lm_train_text "data/${train_set}/text" "$@" 
    
    
