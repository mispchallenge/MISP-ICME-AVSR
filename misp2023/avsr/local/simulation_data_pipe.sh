#!/usr/bin/env bash
export PATH="/yrfs2/cv1/hangchen2/espnet/mispi/avsr:$PATH"
source ./bashrc
# # export cmd=./shared/run.pl
set -eou pipefail
cmd=run.pl
set -e
#######very import, very import !!!! please normalization the torch before you have the simulation process #####
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
stop_stage=$stage

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    #new misp noise
    # cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr/Noise_RIR
    # for subset in train_far_audio dev_far_audio ;do
    #     python  ../local/make_noise_list.py misp_noise/$subset > misp_noise/${subset}_list
    # done
    # cat misp_noise/train_far_audio_list misp_noise/dev_far_audio_list > misp_noise/train_dev_far_audio_list
    
    #misp awk noise
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    for subdir in train;do
        noisesubdir=/yrfs2/cv1/hangchen2/espnet/mispi/avsr/Noise_RIR/mispavwws_noise/$subdir/far
        tgtdir=/yrfs2/cv1/hangchen2/espnet/mispi/avsr/Noise_RIR/mispavwws_noise/data/${subdir}_far
        python ./local/kaldi_datadir_pre_v1.py  --mode ptdir2scp --suffix ".wav" --filename wav.scp --pt_dir $noisesubdir --tgtdir $tgtdir
        awk '$1 ~ /_0_/' $tgtdir/wav.scp > $tgtdir/wav.scp.tmp 
        mv $tgtdir/wav.scp.tmp $tgtdir/wav.scp
        cat $tgtdir/wav.scp | sort -k1 > $tgtdir/wav.scp.tmp && mv $tgtdir/wav.scp.tmp $tgtdir/wav.scp
        ./local/create_shapefile.sh --nj 32 --input "$tgtdir/wav.scp" --output "$tgtdir/speech_shape"
        python  ./local/make_noise_list.py $tgtdir/wav.scp $tgtdir/speech_shape 0.5 > $tgtdir/filter_noise_list
    done
fi

#convert flac2pt pt.int16 use to save them 
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    ptoutdir=/train13/cv1/hangchen2/viseme_based_lipreading/raw_data/MISP2021AVSRv2/train_wave/middle/wpe/gss_new_pt
    python local/convertflac2pt.py '/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_mid/wav.scp' $ptoutdir
    
    #fix gss_train_mid_new
    #1.
    tgtdir=gss_train_mid_new
    cp -r dump/raw/gss_train_mid dump/raw/$tgtdir
    rm dump/raw/$tgtdir/wav.scp
    python ./local/kaldi_datadir_pre_v1.py  --mode ptdir2scp --suffix ".pt" \
        --filename wav.scp --pt_dir $ptoutdir --tgtdir dump/raw/$tgtdir

    #2.
    filter_dir=dump/raw/$tgtdir
    for file in text utt2spk;do 
        if [ -e ${filter_dir}/$file ];then    
            awk '{gsub(/Middle_/, "", $1); print $1 " " $2}' ${filter_dir}/$file > ${filter_dir}/$file.tmp && mv ${filter_dir}/$file.tmp ${filter_dir}/$file
        fi
    done 
    [ -e ${filter_dir}/spk2utt ] && rm ${filter_dir}/spk2utt
        utils/utt2spk_to_spk2utt.pl "$filter_dir/utt2spk"  > "$filter_dir/spk2utt"
    cat $filter_dir/wav.scp | sort -k1 > $filter_dir/wav.scp.tmp
    mv $filter_dir/wav.scp.tmp $filter_dir/wav.scp
    python ./local/create_shapefile.py --input "$filter_dir/text" --output "$filter_dir/text_shape.char"
    ./local/create_shapefile.sh --nj 32 --input "$filter_dir/wav.scp" --output "$filter_dir/speech_shape" --dimnum 1 
    mv dump/raw/gss_train_mid  dump/raw/gss_train_org 
    mv dump/raw/gss_train_mid_new  dump/raw/gss_train
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    _tag_root=dump/raw

    python ./local/addid_datadir.py --srcdirs  $_tag_root/gss_train_far $_tag_root/gss_train_mid $_tag_root/train_near \
        --addids gssfar gssmid near \
        --movefiles wav.scp speech_shape text text_shape.char utt2spk  \
        --tgtdir 'dump/raw/a_combinetmp' \
        --sufix True

    utils/combine_data.sh --skip_fix true --extra-files "speech_shape text_shape.char" $_tag_root/gss_train_all "$_tag_root/a_combinetmp/train_near $_tag_root/a_combinetmp/gss_train_mid $_tag_root/a_combinetmp/gss_train_far"
    utils/utt2spk_to_spk2utt.pl $_tag_root/gss_train_all/utt2spk > $_tag_root/gss_train_all/spk2utt
    python ./local/kaldi_datadir_pre_v1.py --mode "speech2reco2dur" --shape_file ./dump/raw/gss_train_all/speech_shape
    ./local/headsplitdir.sh --save_num 100 --inputdir dump/raw/gss_train_all --outputdir dump/raw/gss_train_all_100
fi

## clean rir
# /yrfs2/cv1/hangchen2/espnet/mispi/avsr/Noise_RIR/BUT_Reverb/rir_list-> filter_smallroom_rir_list
# small_room=/yrfs2/cv1/hangchen2/espnet/mispi/avsr/Noise_RIR/RIRS_NOISES/simulated_rirs/smallroom/rir_list
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr/Noise_RIR
    awk '$2 ~ /C236|E112|L212|L207|Q301/' BUT_Reverb/rir_list > BUT_filter_smallroom_rir_list
    cat BUT_filter_smallroom_rir_list RIRS_NOISES/simulated_rirs/smallroom/rir_list > sim_real_smallroom_rir_list
fi

#simulation (gssgpu_mid + near)
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    ##combine midnearfar
    _tag_root="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw"
    python ./local/addid_datadir.py --srcdirs  $_tag_root/gssgpu_train_far $_tag_root/gssgpu_train_mid $_tag_root/train_near \
        --addids gssgpufar gssgpumid near \
        --movefiles wav.scp speech_shape text text_shape.char utt2spk  \
        --tgtdir 'dump/raw/a_combinetmp' \
        --sufix True

    utils/combine_data.sh --skip_fix true --extra-files "speech_shape text_shape.char" $_tag_root/combine/gssgpu_train_all "$_tag_root/a_combinetmp/train_near $_tag_root/a_combinetmp/gssgpu_train_mid $_tag_root/a_combinetmp/gssgpu_train_far"
    utils/utt2spk_to_spk2utt.pl $_tag_root/combine/gssgpu_train_all/utt2spk > $_tag_root/combine/gssgpu_train_all/spk2utt
    python ./local/kaldi_datadir_pre_v1.py --mode "speech2reco2dur" --shape_file ./dump/raw/combine/gssgpu_train_all/speech_shape

    foreground_snrs="15:10:5:0"
    background_snrs="15:10:5:0"
    num_data_reps=1
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    rvb_opts+=(--rir-set-parameters "1.0, sim_real_smallroom_rir_list")
    rvb_opts+=(--noise-set-parameters "mispavwws_noise/data/train_far/filter_noise_list")
    inputdir=../dump/raw/combine/gssgpu_train_all
    outputdir=data/gssgpu_train_all_BUTReverb_MISPAWKnoise
  ./steps/data/reverberate_data_dir_new.py \
    "${rvb_opts[@]}" \
    --prefix "aug" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications ${num_data_reps} \
    --max-noises-per-minute 1 \
    --source-sampling-rate 16000 \
    --set_max_noises_recording 1 \
    $inputdir $outputdir
    awk '{gsub($2, "python local/read_pt2stream.py"); print}' $outputdir/wav.scp > $outputdir/wav.scp.tmp && mv $outputdir/wav.scp.tmp $outputdir/wav.scp
    nj=128
    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" --audio-format pipeline2pt --fs "16000" \
    "${outputdir}/wav.scp" "dump/raw/org/train8/gssgpu_train_all_BUTReverb_MISPAWKnoise"
fi

#format dataset
# fix rev-S to S yourself in utt2spk and spk to utt in datadir
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    # fix rev-S to S yourself in utt2spk and spk to utt in datadir
    dset=gssgpu_train_all
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    # cp -r /yrfs2/cv1/hangchen2/espnet/mispi/avsr/data/gssgpu_train_all_BUTReverb_MISPAWKnoise  /yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/combine/gssgpu_train_all_BUTReverb_MISPAWKnoise
    mv dump/raw/combine/${dset}_BUTReverb_MISPAWKnoise/wav.scp dump/raw/combine/${dset}_BUTReverb_MISPAWKnoise/wav_org.scp
    for file in wav.scp utt2num_samples;do
        cp dump/raw/org/train8/${dset}_BUTReverb_MISPAWKnoise/$file dump/raw/combine/${dset}_BUTReverb_MISPAWKnoise/$file 
    done
    cp dump/raw/org/train8/${dset}_BUTReverb_MISPAWKnoise/utt2num_samples dump/raw/combine/${dset}_BUTReverb_MISPAWKnoise/speech_shape 
    awk '{print "aug1-" $1,$2}' "dump/raw/combine/${dset}/text_shape.char" > dump/raw/combine/${dset}_BUTReverb_MISPAWKnoise/text_shape.char 
    
    
fi

#httang pdf pre-train 
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    # # create haitao pdf in dump/raw/org pdf_htt
    # tgtdir="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/org/pdf_httang"
    # python local/haitao_pdf.py 
    # for file in dev_far_pdf_httang_9028.scp dev_far_pdf_httang_9028_shape.scp;do
    #         cat $tgtdir/${file} | sort -k1 > $tgtdir/${file}.tmp && mv $tgtdir/${file}.tmp $tgtdir/${file}
    # done
    
    # # # #inner set with audiosp
    dumpdir=dump/raw/a_pdf
    # pdf_dir=dump/raw/org/pdf_httang
    # for subset in "train" "eval";do 
    #     if [[ $subset == "train" ]];then 
    #         a_dir=dump/raw/gss_train_far_sp
    #     else 
    #         a_dir=dump/raw/gss_sum_far
    #     fi
    #     echo $a_dir
    #     subdir=$dumpdir/${subset}_far_far 
    #     pdffilename=${subset}_far_pdf_httang_9028.scp
    #     _tag_set=$subdir
    #     mkdir -p $subdir
    #     cp $pdf_dir/$pdffilename $subdir/$pdffilename
    #     cp $pdf_dir/${subset}_far_pdf_httang_9028_shape.scp $subdir/pdf_shape
    #     echo 25 > $subdir/pdf_fps
    #     for file in speech_shape wav.scp utt2spk;do 
    #         awk '$1 !~/sp/' $a_dir/$file > $subdir/$file
    #     done

    #     cat $_tag_set/pdf_shape | awk '{print $1}' > $_tag_set/temp_uid.tmp
    #     utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/wav.scp | sort -k1 > $_tag_set/wav.scp.tmp
    #     mv $_tag_set/wav.scp.tmp  $_tag_set/wav.scp
    #     cat $_tag_set/wav.scp | awk '{print $1}' > $_tag_set/temp_uid.tmp
    #     linenum=$(cat $_tag_set/speech_shape | wc -l)
    #     echo "before $linenum"
    #     for file in speech_shape utt2spk $pdffilename pdf_shape ;do
    #         utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/$file | sort -k1 > $_tag_set/$file.tmp
    #         [ ! -e $_tag_set/.backup ] && mkdir -p $_tag_set/.backup
    #         rm  $_tag_set/$file
    #         mv $_tag_set/$file.tmp  $_tag_set/$file
    #     done
    #     cp $_tag_set/$pdffilename $_tag_set/pdf.scp
    #     linenum=$(cat $_tag_set/speech_shape | wc -l)
    #     echo "after $linenum"
    #     rm $_tag_set/*.tmp
    #     utils/utt2spk_to_spk2utt.pl $_tag_set/utt2spk > $_tag_set/spk2utt
    # done 

    # #concat
    # python ./local/utt_concat_train.py --datadir $dumpdir/train_far_far --catdatadir $dumpdir/train_far_far_cat2  --pdffilename train_far_pdf_httang_9028.scp --mode pdf
    # utils/utt2spk_to_spk2utt.pl $dumpdir/train_far_far_cat2/utt2spk > $dumpdir/train_far_far_cat2/spk2utt

    # #merge 
    # utils/combine_data.sh --extra-files "speech_shape train_far_pdf_httang_9028.scp pdf_shape" "$dumpdir/train_far_far_addcat2" "$dumpdir/train_far_far $dumpdir/train_far_far_cat2" 
    # echo 25 > $dumpdir/train_far_far_addcat2/pdf_fps

    
    utils/combine_data.sh --extra-files "speech_shape train_far_pdf_httang_9028.scp pdf_shape" "$dumpdir/train_far_far_spcat2" "$dumpdir/train_far_far_sp $dumpdir/train_far_far_cat2" 
    echo 25 > $dumpdir/train_far_far_addcat2/pdf_fps
    
fi

#gpu gss train far 
if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    cd /yrfs2/cv1/hangchen2/espnet/mispi/avsr
    #concat 1.33333 0.8 1.0 gssfar
    speed_perturb_factors="1.33333 0.8"
    train8dir=dump/raw/org/train8
    for dset in train;do
        tgtdir=gss_${dset}_far
        for factor in ${speed_perturb_factors};do 
            # mkdir -p dump/raw/${tgtdir}_sp${factor}
            
            python ./local/kaldi_datadir_pre_v1.py  --mode ptdir2scp --suffix ".pt" \
            --filename wav.scp --pt_dir $train8dir/${tgtdir}_sp${factor} --tgtdir dump/raw/${tgtdir}_sp${factor}

            for file in feats_type spk2utt text utt2spk text utt2num_samples;do
                cp $train8dir/${tgtdir}_sp${factor}/$file dump/raw/${tgtdir}_sp${factor}/$file
            done 
            mv dump/raw/${tgtdir}_sp${factor}/utt2num_samples dump/raw/${tgtdir}_sp${factor}/speech_shape
            python ./local/create_shapefile.py --input "dump/raw/${tgtdir}_sp${factor}/text" --output "dump/raw/${tgtdir}_sp${factor}/text_shape.char"
        done 
    done

    utils/combine_data.sh --extra-files "speech_shape text_shape.char" "dump/raw/combine/gss_train_far_sp0.8+1.0+1.3" "dump/raw/gss_train_far dump/raw/gss_train_far_sp0.8 dump/raw/gss_train_far_sp1.33333"
    

    # create sp cpugss pdfdir
    dumpdir=dump/raw/a_pdf
    pdf_dir=dump/raw/org/pdf_httang
    for subset in "train";do 
        a_dir="dump/raw/combine/gss_train_far_sp0.8+1.0+1.3"
        subdir=$dumpdir/${subset}_far_far_sp 
        pdffilename=${subset}_far_pdf_httang_9028.scp
        _tag_set=$subdir
        mkdir -p $subdir
        cp $pdf_dir/$pdffilename $subdir/$pdffilename
        cp $pdf_dir/${subset}_far_pdf_httang_9028_shape.scp $subdir/pdf_shape
        echo 25 > $subdir/pdf_fps
        for file in speech_shape wav.scp utt2spk;do 
           cp $a_dir/$file  $subdir/$file
        done
        #inner section
        cat $_tag_set/pdf_shape | awk '{print $1}' > $_tag_set/temp_uid.tmp
        utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/wav.scp | sort -k1 > $_tag_set/wav.scp.tmp
        mv $_tag_set/wav.scp.tmp  $_tag_set/wav.scp
        cat $_tag_set/wav.scp | awk '{print $1}' > $_tag_set/temp_uid.tmp
        linenum=$(cat $_tag_set/speech_shape | wc -l)
        echo "before $linenum"
        for file in speech_shape utt2spk $pdffilename pdf_shape ;do
            utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/$file | sort -k1 > $_tag_set/$file.tmp
            [ ! -e $_tag_set/.backup ] && mkdir -p $_tag_set/.backup
            rm  $_tag_set/$file
            mv $_tag_set/$file.tmp  $_tag_set/$file
        done
        cp $_tag_set/$pdffilename $_tag_set/pdf.scp
        linenum=$(cat $_tag_set/speech_shape | wc -l)
        echo "after $linenum"
        rm $_tag_set/*.tmp
        utils/utt2spk_to_spk2utt.pl $_tag_set/utt2spk > $_tag_set/spk2utt
    done     



    # create sp gssgpu pdfdir
    utils/combine_data.sh --extra-files "speech_shape text_shape.char" "dump/raw/combine/gssgpu_train_far_sp0.8+1.0+1.3" "dump/raw/gssgpu_train_far dump/raw/combine/gssgpu_train_far_sp0.8 dump/raw/combine/gssgpu_train_far_sp1.33333"
    dumpdir=dump/raw/a_pdf
    pdf_dir=dump/raw/org/pdf_httang
    for subset in "train" "eval";do 
        if [[ $subset == "train" ]];then 
            a_dir="dump/raw/combine/gssgpu_train_far_sp0.8+1.0+1.3"
            subdir=$dumpdir/gssgpu_${subset}_far_far_sp 
        else
            a_dir="dump/raw/21eval/gssgpu_sum_far"
            subdir=$dumpdir/gssgpu_${subset}_far_far 
        fi
        
        pdffilename=${subset}_far_pdf_httang_9028.scp
        _tag_set=$subdir
        mkdir -p $subdir
    
        # python local/align_id.py --alignfile $pdf_dir/$pdffilename --reffile $a_dir/wav.scp --outfile $subdir/$pdffilename
        python local/align_id.py --alignfile $pdf_dir/${subset}_far_pdf_httang_9028_shape.scp --reffile $a_dir/wav.scp --outfile $subdir/pdf_shape
        echo 25 > $subdir/pdf_fps
        for file in speech_shape wav.scp utt2spk;do 
           cp $a_dir/$file  $subdir/$file
        done
        #inner section
        cat $_tag_set/pdf_shape | awk '{print $1}' > $_tag_set/temp_uid.tmp
        utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/wav.scp | sort -k1 > $_tag_set/wav.scp.tmp
        mv $_tag_set/wav.scp.tmp  $_tag_set/wav.scp
        cat $_tag_set/wav.scp | awk '{print $1}' > $_tag_set/temp_uid.tmp
        linenum=$(cat $_tag_set/speech_shape | wc -l)
        echo "before $linenum"
        for file in speech_shape utt2spk $pdffilename pdf_shape ;do
            utils/filter_scp.pl  $_tag_set/temp_uid.tmp $_tag_set/$file | sort -k1 > $_tag_set/$file.tmp
            [ ! -e $_tag_set/.backup ] && mkdir -p $_tag_set/.backup
            rm  $_tag_set/$file
            mv $_tag_set/$file.tmp  $_tag_set/$file
        done
        cp $_tag_set/$pdffilename $_tag_set/pdf.scp
        linenum=$(cat $_tag_set/speech_shape | wc -l)
        echo "after $linenum"
        rm $_tag_set/*.tmp
        utils/utt2spk_to_spk2utt.pl $_tag_set/utt2spk > $_tag_set/spk2utt
    done 



fi

#combine traindatas
if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    #comebine final training data
    # srcdir=dump/raw/combine/gss_train_all_BUTReverb_MISPAWKnoise
    # tgtdir=dump/raw/combine/gss_train_midnear_BUTReverb_MISPAWKnoise
    # cp -r $srcdir $tgtdir
    # rm $tgtdir/wav_org.scp
    # for file in speech_shape text text_shape.char utt2num_samples utt2spk utt2uniq wav.scp;do
    #     awk '$1 !~/gssfar/' $tgtdir/$file > $tgtdir/$file.tmp && mv $tgtdir/$file.tmp $tgtdir/$file
    # done 
    # utils/utt2spk_to_spk2utt.pl "$tgtdir/utt2spk"  > "$tgtdir/spk2utt"

    #fix gssgpu far
    # simudir=/train13/cv1/hangchen2/viseme_based_lipreading/raw_data/MISP2021AVSRv2/train_wave/far/wpe/gss_new_pt_BUTReverb_MISPAWKnoise
    # tgtdir=dump/raw/combine/gssgpu_train_far_BUTReverb_MISPAWKnoise
    # python ./local/kaldi_datadir_pre_v1.py  --mode ptdir2scp --suffix ".pt" --filename wav.scp --pt_dir $simudir --tgtdir $tgtdir
    # for file in feats_type spk2utt text utt2spk text utt2num_samples;do
    #     cp $simudir/$file $tgtdir/$file
    # done 
    # cp $tgtdir/utt2num_samples $tgtdir/speech_shape
    # python ./local/create_shapefile.py --input "$tgtdir/text" --output "$tgtdir/text_shape.char"

    # utils/combine_data.sh --extra-files "speech_shape text_shape.char" "dump/raw/gssgpu_train_far_sp" "dump/raw/gssgpu_train_far_sp0.9 dump/raw/gssgpu_train_far_sp1.1 dump/raw/gssgpu_train_far"
    ##gssgpu train_sp_cat 
    # utils/combine_data.sh --extra-files "speech_shape text_shape.char" "dump/raw/combine/gssgpu_train_far_sp_cat2" "dump/raw/gssgpu_train_far_sp dump/raw/combine/gssgpu_train_far_cat2"
    ##gssgpu train_sp_cat near_mid 
    # utils/combine_data.sh --extra-files "speech_shape text_shape.char" "dump/raw/combine/gssgpu_train_all_sp_cat2" "dump/raw/gssgpu_train_mid  dump/raw/train_near dump/raw/combine/gssgpu_train_far_sp_cat2"
    ##gssgpu 1k
    # utils/combine_data.sh --extra-files "speech_shape text_shape.char" "dump/raw/combine/gssgpu_train_1k" "dump/raw/combine/gssgpu_train_all_sp_cat2 dump/raw/combine/gss_train_midnear_BUTReverb_MISPAWKnoise dump/raw/combine/gssgpu_train_far_BUTReverb_MISPAWKnoise"
    ##gssgpu 2k

fi







