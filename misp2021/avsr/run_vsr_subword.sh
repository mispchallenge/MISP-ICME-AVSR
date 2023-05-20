#!/usr/bin/env bash
source ./bashrc
set -eou pipefail

train_set=av_pinyin_tone/av_pinyin_tone_3168/train_far_farmidlip 
valid_set=av_pinyin_tone/av_pinyin_tone_3168/dev_far_farmidlip
test_sets=av_pinyin_tone/av_pinyin_tone_3168/dev_far_farmidlip
asr_config=conf/train_vsr.yaml
vsr_exp=exp_vsr/tritone3_lipfarmid_com00_100fps 
gpu_command= #CUAD_VISIBLE_DIVICES=0
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml
use_lm=true
use_word_lm=false
opts=""
if [[ -n $gpu_command ]];then 
    opts+="--gpu_command $gpu_command"
fi


./vsr.sh                                   \
    --stage 4                              \
    --stop_stage 4                         \
    --vsr_exp ${vsr_exp}                   \
    --lang zh                              \
    --nj 8                                 \
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 1                               \
    --train_pdf_fps 100                    \
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
    
    
