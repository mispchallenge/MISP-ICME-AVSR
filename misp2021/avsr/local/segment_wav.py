import os, codecs
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import argparse

def interface_kaldi2segmentsdir(data_dir, output_dir, wavids=None):
    with codecs.open(os.path.join(data_dir, 'wav.scp'), 'r') as handle:
            lines_content = handle.readlines()
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    wav_dic = {}

    for wav_line in wav_lines:
        name, path = wav_line.split(' ')
        if ".wav" not in path:
            path = path + "_0.wav"
        if wavids:
            if name not in wavids :
                continue
        wav_dic[name] = path  
      
    with codecs.open(os.path.join(data_dir, 'segments'), 'r') as handle:
            lines_content = handle.readlines()
    segment_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    wavpath2segment = {}
    for segment_line in segment_lines:
        segment_name, wav_name, start, end = segment_line.split(' ')
        if wav_name:
            if wav_name not in wavids:
                continue
        start, end = float(start), float(end)
        if wav_dic[wav_name] in wavpath2segment:
            wavpath2segment[wav_dic[wav_name]].append([start, end, segment_name])
        else:
            wavpath2segment[wav_dic[wav_name]] = [[start, end, segment_name]]
    # wavpath2segment -> {wav_path:[[start,end,segment_name],[start,end,segment_name],[start,end,segment_name]]}
    overall_wav_path = sorted([*wavpath2segment.keys()])

    for wav_idx in tqdm(range(len(overall_wav_path)), leave=True):
        wav_name = overall_wav_path[wav_idx].split("/")[-1].replace("_0.wav","")
        wavdir = os.path.join(output_dir,wav_name)
        os.makedirs(wavdir,exist_ok=True)
        sample_rate, data = wavfile.read(overall_wav_path[wav_idx])
        for segment_info in wavpath2segment[overall_wav_path[wav_idx]]:
            start = int(round(sample_rate * segment_info[0]))
            end = int(round(sample_rate * segment_info[1]))  
            if start < end <= len(data):
                seg_path = os.path.join(wavdir, segment_info[2] + ".wav")
                segment_wave_data = data[start: end]
                wavfile.write(seg_path,sample_rate,segment_wave_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("segment wav")
    parser.add_argument( "--data_dir", type=str, default=None, help="cluster_cer,cer,result_cer")
    parser.add_argument( "--output_dir", type=str, default=None, help="")
    parser.add_argument( "--wavids", type=str, nargs="+",default=None, help="")
    args = parser.parse_args()
    interface_kaldi2segmentsdir(args.data_dir,args.output_dir,args.wavids)