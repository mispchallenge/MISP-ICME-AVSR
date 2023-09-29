#!/usr/bin/env bash
source ./bashrc #load customized enviroment
set -eou pipefail

train_set=22_eval_far_farlip #ignore
valid_set=22_eval_far_farlip #ignore
test_sets=22_eval_far_farlip #customized kalid-like eval datadir dump/raw/22_eval_far_farlip
avsr_exp=exp/av/AVnewcross_interctc_farmidnear_lipfmid #model_path and the model config file is $avsr_exp/confing.yaml
inference_config=conf/decode_asr.yaml #decode config
use_lm=false #LM is forbidden
use_word_lm=false


./avsr.sh                                  \
    --stage 1                              \
    --stop_stage 2                         \
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
    --inference_config "${inference_config}" \
    --use_word_lm ${use_word_lm}           \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             
    
    
