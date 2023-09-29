#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
source ./bashrc
set -e
set -u
# set -x
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.
gpu_command=
# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3

# Language model related
use_lm=true       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
use_wavaug_preprocessor=false
asr_tag=       # Suffix to the result dir for asr model training.
asr_exp=       # Specify the directory path for ASR experiment.
vsr_exp=
avsr_exp=
               # If this option is specified, asr_tag is ignored.
asr_stats_dir= # Specify the directory path for ASR statistics.
avsr_stats_dir= # Specify the directory path for AVSR statistics.
asr_config=    # Config for asr model training.
asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=utterance_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
k2_ctc_decoding=true
use_nbest_rescoring=true # use transformer-decoder
                         # and transformer language model for nbest rescoring
num_paths=1000 # The 3rd argument of k2.random_paths.
nll_batch_size=100 # Affect GPU memory usage when computing nll
                   # during nbest rescoring
k2_config=./conf/decode_asr_transformer_with_k2.yaml

use_streaming=false # Whether to use streaming decoding

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_asr_model=valid.acc.best.pth # ASR model path for decoding.
                                      # e.g.
                                      # inference_asr_model=train.loss.best.pth
                                      # inference_asr_model=3epoch.pth
                                      # inference_asr_model=valid.acc.best.pth
                                      # inference_asr_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
front_fix=       # to split different vision origianl datadir
self_fix=       # to split different dirs when in the a&v data combination stage
lippos="far"     # to determied the position of lip
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
asr_text_fold_length=150   # fold_length for text data during ASR training.
lm_fold_length=150         # fold_length for LM training.
train_pdf_fps=
pdf2phonemap= #cluster hmm-state to mono phone
help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ASR model related
    --asr_tag          # Suffix to the result dir for asr model training (default="${asr_tag}").
    --asr_exp          # Specify the directory path for ASR experiment.
    --avsr_exp
                       # If this option is specified, asr_tag is ignored (default="${asr_exp}").
    --asr_stats_dir    # Specify the directory path for ASR statistics (default="${asr_stats_dir}").
    --avsr_stats_dir    # Specify the directory path for AVSR statistics (default="${avsr_stats_dir}").
    --asr_config       # Config for asr model training (default="${asr_config}").
    --asr_args         # Arguments for asr model training (default="${asr_args}").
                       # e.g., --asr_args "--max_epoch 10"
                       # Note that it will overwrite args in asr config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_asr_model # ASR model path for decoding (default="${inference_asr_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --use_streaming       # Whether to use streaming decoding (default="${use_streaming}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --asr_speech_fold_length # fold_length for speech data during ASR training (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training (default="${asr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# #generate train_set by prefix and fixname like gss_ + train_far
# train_set=${front_fix}${train_set}
# valid_set=${front_fix}${valid_set}
# test_sets=${front_fix}${test_sets}

# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
     token_list="${chartoken_list}"
    bpemodel=none
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi

# Set tag for naming of model directory
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        asr_tag+="_${lang}_${token_type}"
    else
        asr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
fi

if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${asr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${lang}_${token_type}"
    else
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_stats_dir+="_sp"
    fi
fi

if [ -z "${avsr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        avsr_stats_dir="${expdir}/avsr_stats_${feats_type}_${lang}_${token_type}"
    else
        avsr_stats_dir="${expdir}/avsr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        avsr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        avsr_stats_dir+="_sp"
    fi
fi

if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi

# The directory used for training commands
if [ -z "${asr_exp}" ]; then
    asr_exp="${expdir}/asr_${asr_tag}"
fi
if [ -z "${avsr_exp}" ]; then
    avsr_exp="${expdir}/avsr_${asr_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
      inference_tag+="_k2_ctc_decoding_${k2_ctc_decoding}"
      inference_tag+="_use_nbest_rescoring_${use_nbest_rescoring}"
    fi
fi


# ========================== Main stages start from here. ==========================

    if ! "${skip_data_prep}"; then
        # here we assume that your have finished lip roi detection and general dection jsons, you can find them in misp2021 dataset
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
            #a. segments roi lip using lip rois jsons
            for x in train far dev eval; do
                for roi_type in lip head; do
                    for y in far middle; do
                        if [[ $y = far ]]; then
                        need_speaker=true
                        else
                        need_speaker=false
                        fi
                        local/prepare_roi.sh --python_path $python_path  --cuda_index 6 --local true --nj 2 --roi_type $roi_type --drop_threshold 0.0 --roi_size "96 96" --need_speaker $need_speaker \
                        avsrdata/${x}_${y}_video avsrreleased_data/misp2021_avsr/${x}_${y}_detection_result avsrfeature/misp2022_avsr/${x}_${y}_video_${roi_type}_segment
                done
                done
            done

            #b. roi.scp in dump/raw/org/rois
            log "stage 0 create roi.scp,pdf.scp in dump/org/rois \ dump/org/ pos={far,middle} setclass={train,dev,sum}"
            # pt_dir is the path to store rois in roi_pt.scp (.scp: <uid> roi.pt)
            ./local/extract_lip_roi.sh  --pt_dir misp2021_avsr/feature/misp2021_avsr \
                                        --tagdir dump/raw/org/rois

            #c. pdf.scp in dump/raw/org/pdf_pinyin_tone
            # here we assume you have trained an GMM-HMM model and PDFs using force alignment, related code you can refer to  https://github.com/mispchallenge/MISP2021-AVSR
            for setclass in train dev;do
                tagdir=dump/raw/org/pdf_pinyin_tone
                for pdf_sufix in pinyin-tone_nos3s5p0_4gram_l8000g64000  pinyin-tone_nos3s5p0_4gram_l4000g32000;do
                    subptdir=feature/${setclass}_pdf/tri3/far/wpe/gss/$pdf_sufix
                    grain=tri
                    mkdir -p $tagdir
                    if [ -e $subptdir/num_pdf ];then 
                        pdfcnum=$(cat $subptdir/num_pdf)
                    else
                        pdfcnum=$pdfcnum
                    fi
                    pdf_fname=${setclass}_pdf_${grain}_${pdfcnum}
                    echo "$pdf_fname"
                    python local/lip_roi.py --pt_dir $subptdir/pt --roiscpdir ${tagdir} --filename ${pdf_fname}.scp
                    cat ${tagdir}/${pdf_fname}.scp | sort -k1 > ${tagdir}/${pdf_fname}_tmp.scp
                    rm  ${tagdir}/${pdf_fname}.scp
                    mv ${tagdir}/${pdf_fname}_tmp.scp ${tagdir}/${pdf_fname}.scp
                done
            done

        fi
        
        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 

            log "stage 1 create train avsrdir in ${data_feats}/av"
            _audio_pos=far
            _stage=0
            for _lip_pos in far middle ;do 
                for setclass in train dev sum ;do 
                    _sub_tag_set="${setclass}_${_audio_pos}_${_lip_pos}lip"
                    _tag_set="${data_feats}/av/${setclass}_${_audio_pos}_${_lip_pos}lip"
                    _srcdir="${data_feats}/gss_${setclass}_${_audio_pos}"
                    if [[ $setclass == "train" ]];then 
                            _srcdir=${data_feats}/gss_${setclass}_${_audio_pos}_sp 
                        fi
                    if [ ${_stage} -le 0 ];then
                        log "preparing audio dir"

                        python ./local/addid_datadir.py --srcdirs $_srcdir   \
                        --addids "kong" \
                        --movefiles wav.scp speech_shape text text_shape.char utt2spk \
                        --tgtdir "${data_feats}/av" \
                        --cutsp true \
                        --tgtsubdirs $_sub_tag_set || exit 1;
                        cp $_srcdir/feats_type "${_tag_set}"
                    fi   
                    if [ ${_stage} -le 1 ];then
                        log "preparing roi_shape"
                        cp  ${data_feats}/org/rois/${setclass}_${_lip_pos}_roi.scp  ${_tag_set}/roi.scp
                        ./local/create_shapefile.sh  --nj 16 --input "${_tag_set}/roi.scp" --output "${_tag_set}/video_shape" --dimnum 4
                        # #c.fix roi.scp and audio dir
                        log "fix ${_tag_set}"
                        #c.1get temp_uid.tmp which is the intersection of roi.scp and wav.scp
                        cat $_tag_set/roi.scp | awk '{print $1}' > $_tag_set/temp_uid.tmp
                        utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/wav.scp > $_tag_set/wav.scp.tmp
                        mv $_tag_set/wav.scp.tmp  $_tag_set/wav.scp
                        cat $_tag_set/wav.scp | awk '{print $1}' > $_tag_set/temp_uid.tmp
                    fi
                    if [ ${_stage} -le 2 ];then
                        log "preparing avsr datadir"
                        linenum=$(cat $_tag_set/speech_shape | wc -l)
                        echo "before $linenum"
                        for file in speech_shape roi.scp video_shape text text_shape.char utt2spk;do
                            utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/$file > $_tag_set/$file.tmp
                            [ ! -e $_tag_set/.backup ] && mkdir -p $_tag_set/.backup
                            mv $_tag_set/$file  $_tag_set/.backup/$file
                            mv $_tag_set/$file.tmp  $_tag_set/$file
                        done
                        linenum=$(cat $_tag_set/speech_shape | wc -l)
                        echo "after $linenum"
                        rm $_tag_set/*.tmp
                        utils/fix_data_dir.sh $_tag_set
                    fi
                    cp $_srcdir/spk2utt $_tag_set/spk2utt
                    cp $_srcdir/utt2spk $_tag_set/utt2spk
                    utils/fix_data_dir.sh $_tag_set
                done
            done

        fi

        if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then         
            log "stage 2 comebine far&midlip avsrdir in ${data_feats}/av"
            python ./local/addid_datadir.py --srcdirs  dump/raw/av/train_far_farlip dump/raw/av/train_far_middlelip \
                    --addids farlip midlip \
                    --movefiles roi.scp video_shape wav.scp speech_shape text text_shape.char utt2spk \
                    --tgtdir 'dump/raw/av/combinetmp'
            mkdir -p dump/raw/av/combinetmp
            utils/combine_data.sh --skip_fix true --extra-files "roi.scp video_shape speech_shape text_shape.char" "dump/raw/av/train_far_farmidlip" "dump/raw/av/combinetmp/train_far_farlip dump/raw/av/combinetmp/train_far_middlelip"
            utils/utt2spk_to_spk2utt.pl dump/raw/av/train_far_farmidlip/utt2spk > dump/raw/av/train_far_farmidlip/spk2utt
            cp dump/raw/av/train_far_farlip/feats_type dump/raw/av/train_far_farmidlip
            rm -rf dump/raw/av/combinetmp

        fi

        if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
            _stage=0
            _stop_stage=0
            if [ $_stage -le 0 ] && [ $_stop_stage -ge 0 ];then 
                log "stage 3_0 create train/dev avpdf datadir in ${data_feats}/avpdf based on ${data_feats}/av"
                counter=0
                begin_counter=0
                pdf_fps=25
                grainclass=pinyin_tone
                for subclass in train dev;do 
                    glob=${data_feats}/org/pdf_${grainclass}/${subclass}*tri3*.scp
                    pdffiles=$(find $glob)
                    for pdffile in $pdffiles;do # for each pdf_train/dev_{num}.scp
                        counter=$(bc <<< "${counter}+1")
                        if [ $counter -le $begin_counter ];then 
                            continue 
                        fi
                        pdfcnum=${pdffile##*_}
                        pdfcnum=${pdfcnum%%.*}
                        subavpdf_dir=${data_feats}/av_${grainclass}/av_${grainclass}_${pdfcnum}
                        mkdir -p ${subavpdf_dir}
                        for lippos in far middle;do # for each far/mid lip
                            avsubdir=${subclass}_far_${lippos}lip
                            _tag_set=${subavpdf_dir}/$avsubdir

                            #1. copy fmid audio & roi.scp & pdf.scp to dir and create pdf_shape
                            log "creating ${_tag_set}"
                            if [ ! -f $_tag_set ];then
                                cp -R ${data_feats}/av/$avsubdir ${subavpdf_dir}/
                            fi
                            cp ${data_feats}/org/pdf_${grainclass}/${subclass}_*_${pdfcnum}.scp $_tag_set
                            pdfile=$(find $_tag_set/${subclass}_*_${pdfcnum}.scp)
                            pdffilename=${pdfile##*/}
                            ./local/create_shapefile.sh  --nj 16 --input "${pdfile}" --output "$_tag_set/pdf_shape" --pdfflag true
                            cp 

                            #2.get temp_uid.tmp which is the intersection of roi.scp and wav.scp
                            log "fix ${_tag_set}"
                            cat $_tag_set/pdf_shape | awk '{print $1}' > $_tag_set/temp_uid.tmp
                            utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/wav.scp | sort -k1 > $_tag_set/wav.scp.tmp
                            mv $_tag_set/wav.scp.tmp  $_tag_set/wav.scp
                            cat $_tag_set/wav.scp | awk '{print $1}' > $_tag_set/temp_uid.tmp
                            linenum=$(cat $_tag_set/speech_shape | wc -l)
                            echo "before $linenum"
                            for file in speech_shape roi.scp video_shape text text_shape.char utt2spk $pdffilename pdf_shape ;do
                                utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/$file | sort -k1 > $_tag_set/$file.tmp
                                [ ! -e $_tag_set/.backup ] && mkdir -p $_tag_set/.backup
                                mv $_tag_set/$file  $_tag_set/.backup/$file
                                mv $_tag_set/$file.tmp  $_tag_set/$file
                            done
                            cp $_tag_set/$pdffilename $_tag_set/pdf.scp
                            linenum=$(cat $_tag_set/speech_shape | wc -l)
                            echo "after $linenum"
                            rm $_tag_set/*.tmp
                            utils/utt2spk_to_spk2utt.pl $_tag_set/utt2spk > $_tag_set/spk2utt
                            utils/fix_data_dir.sh $_tag_set
                            echo $pdf_fps > $_tag_set/pdf_fps

                        done
                        
                        #3. combine farlip with midlip
                        log "creating $subavpdf_dir/${subclass}_far_farmidlip "
                        python ./local/addid_datadir.py --srcdirs  $subavpdf_dir/${subclass}_far_farlip $subavpdf_dir/${subclass}_far_middlelip \
                            --addids farlip midlip \
                            --movefiles roi.scp video_shape wav.scp speech_shape text text_shape.char $pdffilename pdf_shape utt2spk  \
                            --tgtdir dump/raw/av_${grainclass}/combinetmp
                        utils/combine_data.sh --skip_fix true --extra-files "roi.scp video_shape speech_shape text_shape.char $pdffilename pdf_shape" "$subavpdf_dir/${subclass}_far_farmidlip" "dump/raw/av_${grainclass}/combinetmp/${subclass}_far_farlip dump/raw/av_${grainclass}/combinetmp/${subclass}_far_middlelip"
                        cp $subavpdf_dir/${subclass}_far_farlip/feats_type $subavpdf_dir/${subclass}_far_farmidlip
                        rm -rf dump/raw/av_${grainclass}/combinetmp/*
                        utils/utt2spk_to_spk2utt.pl $subavpdf_dir/${subclass}_far_farmidlip/utt2spk > $subavpdf_dir/${subclass}_far_farmidlip/spk2utt
                        echo $pdf_fps > $subavpdf_dir/${subclass}_far_farmidlip/pdf_fps
                        ./local/headsplitdir.sh --save_num 100 --inputdir $subavpdf_dir/${subclass}_far_farmidlip --outputdir $subavpdf_dir/${subclass}_far_farmidlip_100
                        echo $pdf_fps > $subavpdf_dir/${subclass}_far_farmidlip_100/pdf_fps
                    done 
                done
                
            fi

        fi
  
    fi

    # visual frontend by correlating lipshape and syllable subwords  
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _vsr_train_dir="${data_feats}/${train_set}"
        _vsr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: VSR Training which is special for force align pdf-av training : train_set=${_vsr_train_dir}, valid_set=${_vsr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi


        _trainpdfname=$(find $_vsr_train_dir/*pdf*.scp)
        _trainpdfname=${_trainpdfname##*/}
        _validpdfname=$(find $_vsr_valid_dir/*pdf*.scp)
        _validpdfname=${_validpdfname##*/}
        _train_pdf_cnum=${_trainpdfname##*_}
        _train_pdf_cnum=${_train_pdf_cnum%%.*}
        _valid_pdf_cnum=${_validpdfname##*_}
        _valid_pdf_cnum=${_valid_pdf_cnum%%.*}
      
     
        if [ $_train_pdf_cnum != $_valid_pdf_cnum ];then 
            log "_train_pdf_cnum:$_train_pdf_cnum but _valid_pdf_cnum:$_valid_pdf_cnum"
            exit 1
        fi

        if [ $pdf2phonemap == "true" ];then 
            _vsr_pdf2phonemap=${_vsr_train_dir%/*}/pdf2phone_map.json
            _opts+="--pdf2phonemap $_vsr_pdf2phonemap "
            _train_pdf_cnum=$(python3 -c "import json; f=open('${_vsr_pdf2phonemap}');dict=json.load(f);value = max([int(value) for value in dict.values() ]);print(value+1)")
        fi

        if [ -z $train_pdf_fps ];then 
            _train_pdf_fps="$(<${_vsr_train_dir}/pdf_fps)"
            _valid_pdf_fps="$(<${_vsr_valid_dir}/pdf_fps)"
            if [ $_train_pdf_fps != $_valid_pdf_fps ];then 
                log "_train_pdf_fps:$_train_pdf_cnum but _valid_pdf_fps:$_valid_pdf_cnum"
                exit 1
            fi
            _opts+="--pdf_fps $_train_pdf_fps --pdf_cnum $_train_pdf_cnum "
        else
            _opts+="--pdf_fps $train_pdf_fps --pdf_cnum $_train_pdf_cnum "
        fi
        
        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.
            mkdir -p ${_vsr_train_dir}/split
            _split_dir="${_vsr_train_dir}/split/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                    --scps \
                        "${_vsr_train_dir}/text" \
                        "${_vsr_train_dir}/$_trainpdfname" \
                        "${_vsr_train_dir}/roi.scp" \
                        "${_vsr_train_dir}/text_shape.${token_type}" \
                        "${_vsr_train_dir}/pdf_shape" \
                        "${_vsr_train_dir}/video_shape" \
                    --num_splits "${num_splits_asr}" \
                    --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/roi.scp,video,pt "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/$_trainpdfname,pdf,pt "
            _opts+="--train_shape_file ${_split_dir}/video_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--train_shape_file ${_split_dir}/pdf_shape "
            
      
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_vsr_train_dir}/roi.scp,video,pt "
            _opts+="--train_data_path_and_name_and_type ${_vsr_train_dir}/$_trainpdfname,pdf,pt "
            _opts+="--train_data_path_and_name_and_type ${_vsr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${_vsr_train_dir}/video_shape "
            _opts+="--train_shape_file ${_vsr_train_dir}/pdf_shape "
            _opts+="--train_shape_file ${_vsr_train_dir}/text_shape.${token_type} "
            _opts+="--valid_data_path_and_name_and_type ${_vsr_valid_dir}/roi.scp,video,pt " 
            _opts+="--valid_data_path_and_name_and_type ${_vsr_valid_dir}/$_validpdfname,pdf,pt "
            _opts+="--valid_data_path_and_name_and_type ${_vsr_valid_dir}/text,text,text " 
            _opts+="--valid_shape_file ${_vsr_valid_dir}/video_shape " 
            _opts+="--valid_shape_file ${_vsr_valid_dir}/pdf_shape " 
            _opts+="--valid_shape_file ${_vsr_valid_dir}/text_shape.${token_type} " 
            _opts+="--token_type ${token_type} " 
            _opts+="--token_list ${token_list} "
    
        fi

        log "Generate '${vsr_exp}/run.sh'. You can resume the process from stage 13 using this script"
        mkdir -p "${vsr_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${vsr_exp}/run.sh"; chmod +x "${vsr_exp}/run.sh" || echo "acl erro skill chmod a+x"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "VSR training started... log: '${vsr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${vsr_exp})"
        else
            jobname="${vsr_exp}/train.log"
        fi
        
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${vsr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --gpu_command "${gpu_command}" \
            --init_file_prefix "${vsr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.vsr_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --resume true \
                --output_dir "${vsr_exp}" \
                ${_opts} ${asr_args}

    fi

    # visual frontend by continuous lipreading recognization
    # remember to set train_pdf_fps
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _vsr_train_dir="${data_feats}/${train_set}"
        _vsr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: VSR Training: train_set=${_vsr_train_dir}, valid_set=${_vsr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _feats_type="$(<${_vsr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((asr_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${asr_speech_fold_length}"
            _input_size="$(<${_vsr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${_vsr_train_dir}/feats_stats.npz "
        fi

        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.
            mkdir -p ${_vsr_train_dir}/split
            _split_dir="${_vsr_train_dir}/split/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                    --scps \
                        "${_vsr_train_dir}/roi.scp" \
                        "${_vsr_train_dir}/text" \
                        "${_split_dir}/video_shape" \
                        "${_vsr_train_dir}/text_shape.${token_type}" \
                    --num_splits "${num_splits_asr}" \
                    --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/roi.scp,video,pt "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/video_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_vsr_train_dir}/roi.scp,video,pt "
            _opts+="--train_data_path_and_name_and_type ${_vsr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${_vsr_train_dir}/video_shape "
            _opts+="--train_shape_file ${_vsr_train_dir}/text_shape.${token_type} "
        fi

        log "Generate '${vsr_exp}/run.sh'. You can resume the process from stage 14 using this script"
        mkdir -p "${vsr_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${vsr_exp}/run.sh"; chmod +x "${vsr_exp}/run.sh" || echo "acl erro skill chmod a+x"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "VSR training started... log: '${vsr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${vsr_exp})"
        else
            jobname="${vsr_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${vsr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --gpu_command "${gpu_command}" \
            --init_file_prefix "${vsr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.vsr_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_vsr_valid_dir}/roi.scp,video,pt" \
                --valid_data_path_and_name_and_type "${_vsr_valid_dir}/text,text,text" \
                --valid_shape_file "${_vsr_valid_dir}/text_shape.${token_type}" \
                --valid_shape_file "${_vsr_valid_dir}/video_shape" \
                --resume true \
                --fold_length "${_fold_length}" \
                --fold_length "${asr_text_fold_length}" \
                --output_dir "${vsr_exp}" \
                ${_opts} ${asr_args}

    fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Decoding: training_dir=${avsr_exp}"
        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi
        if "${use_ngram}"; then
             _opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
        fi

        # 2. Generate run.sh
        log "Generate '${avsr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 15 using this script"
        mkdir -p "${avsr_exp}/${inference_tag}"; echo "${run_args} --stage 15 \"\$@\"; exit \$?" > "${avsr_exp}/${inference_tag}/run.sh"; chmod +x "${avsr_exp}/${inference_tag}/run.sh" || echo "acl erro skill chmod a+x"
        if "${use_k2}"; then
          # Now only _nj=1 is verified if using k2
          asr_inference_tool="espnet2.bin.asr_inference_k2"

          _opts+="--is_ctc_decoding ${k2_ctc_decoding} "
          _opts+="--use_nbest_rescoring ${use_nbest_rescoring} "
          _opts+="--num_paths ${num_paths} "
          _opts+="--nll_batch_size ${nll_batch_size} "
          _opts+="--k2_config ${k2_config} "
        else
          if "${use_streaming}"; then
              asr_inference_tool="espnet2.bin.asr_inference_streaming"
          else
              asr_inference_tool="espnet2.bin.avsr_inference"
          fi
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${avsr_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            if "${use_k2}"; then
              # Now only _nj=1 is verified if using k2
              _nj=1
            else
              _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            fi

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}
            

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                ${python} -m ${asr_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,pt" \
                    --data_path_and_name_and_type "${_data}/roi.scp,video,pt" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --asr_train_config "${avsr_exp}"/config.yaml \
                    --asr_model_file "${avsr_exp}"/"${inference_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                  for i in $(seq "${_nj}"); do
                      cat "${_logdir}/output.${i}/1best_recog/${f}"
                  done | sort -k1 >"${_dir}/${f}"
                fi
            done
        done
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Scoring"
        if [ "${token_type}" = phn ]; then
            log "Error: Not implemented for token_type=phn"
            exit 1
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${avsr_exp}/${inference_tag}/${dset}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"


                elif [ "${_type}" = cer ]; then
                    # Tokenize text to char level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                elif [ "${_type}" = ter ]; then
                    # Tokenize text using BPE
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                fi

                sclite \
		    ${score_opts} \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done
        done

        # [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${avsr_exp}"

        # Show results in Markdown syntax
        scripts/utils/show_asr_result.sh "${avsr_exp}" > "${avsr_exp}"/RESULTS.md
        cat "${avsr_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
