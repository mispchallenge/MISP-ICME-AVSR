#!/usr/bin/env bash
# transform misp data to kaldi format

set -e -o pipefail
echo "$0 $@"
nj=1
need_channel=
stage=0
channel_dir=
. ./cmd.sh || exit 1
. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1


# if [ $# != 4 ]; then
#   echo "Usage: $0 <original-corpus-data-dir> <enhancement-audio-dir> <data-set>  <store-dir>"
#   echo " $0 /path/misp /path/misp_gss train data/gss_train_far"
#   exit 1;
# fi


gss_dir=$1
video_dir=$2
transcription_dir=$3
data_type=$4
store_dir=$5


# wav.scp segments text_sentence utt2spk
# for example: python local/prepare_gss_data.py -nj $nj --without_mp4 True /raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/eval_far_audio_wpe_beamformit/wav /raw7/cv1/hangchen2/misp2021_avsr/released_data/misp2021_avsr/addition_far_video/mp4 /raw7/cv1/hangchen2/misp2021_avsr/released_data/misp2021_avsr/eval_near_transcription/TextGrid eval data/beam_eval_far
# you can change --without_mp4,enhancement_wav,video_path to combine different audio and video filed as you like

if [ ${stage} -le 1 ];then
    _opt=""
    echo "prepare wav.scp segments text_sentence utt2spk"
    echo "$need_channel"
    [[ -n $need_channel ]] && _opts+="--channel_dir $channel_dir --without_wav True"
    python local/prepare_gss_data.py -nj  $nj $_opts $gss_dir $video_dir $transcription_dir $data_type $store_dir

fi

#fix kaldi data dir
if [ ${stage} -le 2 ];then
    for file in wav.scp channels.scp mp4.scp segments utt2spk text_sentence;do
        if [ -f $store_dir/temp/$file ];then
            if [ $file == "text_sentence" ];then 
                cat $store_dir/temp/$file | sort -k 1 | uniq > $store_dir/text 
            else
                cat $store_dir/temp/$file | sort -k 1 | uniq > $store_dir/$file
            fi
        fi
    done
    
    [[ -e $store_dir/temp/channels.scp ]] && cp $store_dir/channels.scp $store_dir/wav.scp
    rm -r $store_dir/temp
    echo "prepare done"

    # generate spk2utt and nlsyms
    utils/utt2spk_to_spk2utt.pl $store_dir/utt2spk | sort -k 1 | uniq > $store_dir/spk2utt
    touch data/nlsyms.txt
    
    utils/fix_data_dir.sh $store_dir

fi

echo "local/prepare_gss_data.sh succeeded"
exit 0