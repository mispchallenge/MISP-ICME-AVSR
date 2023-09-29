import argparse
from distutils.util import strtobool
import os
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from pathlib import Path
import subprocess
from typing import Dict
from typing import List
from typing import Union
from espnet2.fileio.read_text import read_2column_text
from espnet2.fileio.datadir_writer import DatadirWriter
import shutil
def same_distance(a, b):
    if a == b:
        return 0
    else:
        return 1


def naive_loss(multi_unit, new_unit):
    # naive loss for alignment
    error = 0
    for unit in multi_unit:
        if unit != new_unit:
            error += 1
    return error / len(multi_unit)


numbers = "0123456789"


def load_hyp_dict(filename):
    # for each file, it contains decoded results like "uttid trans"
    with open(filename, "r", encoding="utf-8") as test_file:
        result = {}
        while True:
            hyp_line = test_file.readline()
            if not hyp_line:
                break
            hyp_array = hyp_line.strip().split()
            if len(hyp_array) == 2:
                result[hyp_array[0]] = hyp_array[1].replace("<sos/eos>","")
            else:
                result[hyp_array[0]] = ""
    return result


class Distance(object):
    def __init__(self, delimiter, loss_fn, space2str=True):
        self.loss_fn = loss_fn
        self.delimiter = delimiter
        self.space2str = space2str
        self.INS, self.DEL, self.FOR = 1, 2, 3

    def multi_loss(self, multi_unit, new_unit):
        if self.loss_fn is not None:
            return self.loss_fn(multi_unit, new_unit)
        error = 0
        for unit in multi_unit:
            if unit != new_unit:
                error += 1

        return error / len(multi_unit)

    def alignment(self, hyp, ref, norm=False, return_er=False):
        assert type(ref) == str and type(hyp) == str

        if self.delimiter != "":
            ref = ref.split(self.delimiter)
            hyp = hyp.split(self.delimiter)

        ref_len = len(ref)
        hyp_len = len(hyp)

        workspace = np.zeros((len(hyp) + 1, len(ref) + 1))
        paths = np.zeros((len(hyp) + 1, len(ref) + 1))

        for i in range(1, len(hyp) + 1):
            workspace[i][0] = i
            paths[i][0] = self.INS
        for i in range(1, len(ref) + 1):
            workspace[0][i] = i
            paths[0][i] = self.DEL

        for i in range(1, len(hyp) + 1):
            for j in range(1, len(ref) + 1):
                if hyp[i - 1] == ref[j - 1]:
                    workspace[i][j] = workspace[i - 1][j - 1]
                    paths[i][j] = self.FOR
                else:
                    ins_ = workspace[i - 1][j] + 1
                    del_ = workspace[i][j - 1] + 1
                    sub_ = workspace[i - 1][j - 1] + self.loss_fn(
                        hyp[i - 1], ref[j - 1]
                    )
                    if ins_ < min(del_, sub_):
                        workspace[i][j] = ins_
                        paths[i][j] = self.INS
                    elif del_ < min(ins_, sub_):
                        workspace[i][j] = del_
                        paths[i][j] = self.DEL
                    else:
                        workspace[i][j] = sub_
                        paths[i][j] = self.FOR

        i = len(hyp)
        j = len(ref)
        aligned_hyp = []
        aligned_ref = []
        while i >= 0 and j >= 0:
            if paths[i][j] == self.FOR:
                aligned_hyp.append(hyp[i - 1])
                aligned_ref.append(ref[j - 1])
                i = i - 1
                j = j - 1
            elif paths[i][j] == self.DEL:
                aligned_hyp.append("*")
                aligned_ref.append(ref[j - 1])
                j = j - 1
            elif paths[i][j] == self.INS:
                aligned_hyp.append(hyp[i - 1])
                aligned_ref.append("*")
                i = i - 1
            if i == 0 and j == 0:
                i -= 1
                j -= 1

        aligned_hyp = aligned_hyp[::-1]
        aligned_ref = aligned_ref[::-1]

        if self.space2str:
            for i in range(len(aligned_hyp)):
                if aligned_hyp[i] == " ":
                    aligned_hyp[i] = "<space>"
            for i in range(len(aligned_ref)):
                if aligned_ref[i] == " ":
                    aligned_ref[i] = "<space>"

        # print("CER: {}".format(workspace[-1][-1]))
        if return_er:
            return workspace[-1][-1] / len(ref)
        return aligned_hyp, aligned_ref

    def skeleton_align(self, skeleton, align):
        if type(skeleton) == str:
            # detect raw text, format skeleton
            new_skeleton = []
            if self.delimiter != "":
                skeleton = skeleton.split(self.delimiter)
            for unit in skeleton:
                new_skeleton.append([unit])
            skeleton = new_skeleton

        if self.delimiter != "":
            align = align.split(self.delimiter)

        skeleton_len = len(skeleton)
        align_len = len(align)

        workspace = np.zeros((align_len + 1, skeleton_len + 1))
        paths = np.zeros((align_len + 1, skeleton_len + 1))

        for i in range(1, align_len + 1):
            workspace[i][0] = i
            paths[i][0] = self.INS
        for i in range(1, skeleton_len + 1):
            workspace[0][i] = i
            paths[0][i] = self.DEL

        for i in range(1, align_len + 1):
            for j in range(1, skeleton_len + 1):
                ins_ = workspace[i - 1][j] + self.multi_loss(
                    ["*"] * len(skeleton[0]), align[i - 1]
                )
                del_ = workspace[i][j - 1] + self.multi_loss(skeleton[j - 1], "*")
                sub_ = workspace[i - 1][j - 1] + self.multi_loss(
                    skeleton[j - 1], align[i - 1]
                )
                if ins_ < min(del_, sub_):
                    workspace[i][j] = ins_
                    paths[i][j] = self.INS
                elif del_ < min(ins_, sub_):
                    workspace[i][j] = del_
                    paths[i][j] = self.DEL
                else:
                    workspace[i][j] = sub_
                    paths[i][j] = self.FOR
      
        i = align_len
        j = skeleton_len
        new_skeleton = []
        while i >= 0 and j >= 0:
            if paths[i][j] == self.FOR:
                new_skeleton.append(skeleton[j - 1] + [align[i - 1]])
                i = i - 1
                j = j - 1
            elif paths[i][j] == self.DEL:
                new_skeleton.append(skeleton[j - 1] + ["*"])
                j = j - 1
            elif paths[i][j] == self.INS:
                new_skeleton.append(["*"] * len(skeleton[0]) + [align[i - 1]])
                i = i - 1
            if i == 0 and j == 0:
                i -= 1
                j -= 1

        new_skeleton = new_skeleton[::-1]

        return new_skeleton

    def process_skeleton_count(self, skeleton, weight=None, word=False):
        if weight is None:
            weight = [1 / len(skeleton[0])] * len(skeleton[0])

        result = []
        for multi_unit in skeleton:
            temp_dict = {}
            for i in range(len(multi_unit)):
                temp_dict[multi_unit[i]] = temp_dict.get(multi_unit[i], 0) + weight[i]
            result.append(max(temp_dict.items(), key=(lambda item: item[1]))[0])

        return " ".join(result) if word else "".join(result)

    def save_skeleton(self, skeleton):
        num_hyp = len(skeleton[0])
        aligned_string = []
        for i in range(num_hyp):
            aligned_string.append("")
        for multi_unit in skeleton:
            for i in range(num_hyp):
                aligned_string[i] += multi_unit[i]
        return "@@@".join(aligned_string)


def combine(hyp_list, loss, word=False, weight=None):
    # add weight for different weight of voting process
    if word:
        delimiter = " "
    else:
        delimiter = ""
    dist_char = Distance(delimiter=delimiter, loss_fn=loss, space2str=True)

    assert len(hyp_list) > 1
    skeleton = hyp_list[0]

    for align in hyp_list[1:]:
        skeleton = dist_char.skeleton_align(skeleton, align)

    return dist_char.process_skeleton_count(skeleton, weight, word)


def generate_alignment(hyp_list, loss, word=False):
    if word:
        delimiter = " "
    else:
        delimiter = ""
    dist_char = Distance(delimiter=delimiter, loss_fn=loss, space2str=True)

    assert len(hyp_list) > 1
    skeleton = hyp_list[0]

    for align in hyp_list[1:]:
        skeleton = dist_char.skeleton_align(skeleton, align)

    return dist_char.save_skeleton(skeleton)


def check_keys(key, source_list):
    for source in source_list:
        if key not in source.keys():
            return False
    return True

def test():
    test = ["abcd", "bcdeeeee", "bfeeeee"]
    dist_char = Distance(delimiter="", loss_fn=same_distance, space2str=True)
    print(combine(test,naive_loss))

def rover(source_list,outputfile,loss,save_align=False,wordmode=True):
    if len(source_list) == 1:
        shutil.copy(source_list[0],outputfile)
        return None
    source_dicts = []
    for i in range(len(source_list)):
        source_dicts.append(load_hyp_dict(source_list[i]))

    result_file = open(outputfile, "w", encoding="utf-8")

    for key in source_dicts[0].keys():
        final = ""
        if not check_keys(key, source_dicts):
            print("missing key {} in source".format(key))
        hyp_list = []
        for source in source_dicts:
            if key in source.keys() and source[key] != "":
                hyp_list.append(source[key])

        if len(hyp_list) == 0:
            print("all hyps are empty for utt {}".format(key))
            result_file.write("{} {}\n".format(key, final))
            continue
        if len(hyp_list) == 1:
            print("only have one unempty hyp for utt {}".format(key))
            result_file.write("{} {}\n".format(key, final))
            continue

        if save_align:
            final = generate_alignment(hyp_list, loss, wordmode)
            result_file.write("{} {}\n".format(key, final))
        else:
            final = combine(hyp_list, loss, wordmode)
            final = final.replace("*", "")
            result_file.write("{} {}\n".format(key, final))
        # print(key, final)
    result_file.close()

def cer(hyppath,refpath):
    return_lines = []
    scheduler_order = f"./local/cer.sh --ref {refpath} --hyp {hyppath}"
    return_info = subprocess.Popen(scheduler_order, shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    for next_line in return_info.stdout:
        return_line = next_line.decode("utf-8", "ignore")
        if "WER" in return_line:
            return return_line[5:10]

def createchar(refpath):
    refpath = Path(refpath)
    dir = refpath.parent
    tokenpath = dir / (refpath.stem + ".char")
    refdic = read_2column_text(refpath)
    with DatadirWriter(dir) as writer:  
        subwriter = writer[(refpath.stem + ".char")]
        for k,line in refdic.items():
            charlist = [ char for char in line.strip()]
            newline = (" ").join(charlist)
            subwriter[k] = newline
    return str(tokenpath)

def deal_with_result_decode_and_text(result_decode_path,output_path_from_rd):
        content_result_decode = []
        with open(result_decode_path, 'r') as f_rd:
            for line in f_rd.readlines():
                line = line.strip()
                content_result_decode.append(line)
        content_result_decode = sorted(content_result_decode)
        output_from_result_decode = []
        for line in content_result_decode:
            file_name = "_".join(line.split(" ")[0].split("_")[:-1])
            if len(output_from_result_decode) == 0 or output_from_result_decode[-1].split(" ")[0] != file_name:
                output_from_result_decode.append(" ".join([file_name] + line.split(" ")[1:]))
            else:
                output_from_result_decode[-1] += " "+" ".join(line.split(" ")[1:])
        with open(output_path_from_rd, 'w') as f_ord:
            f_ord.write("\n".join(sorted(output_from_result_decode))+"\n")

def add_extra_file(rover_text,extra_text):
    rover_dic,extra_dic =  read_2column_text(rover_text),read_2column_text(extra_text)
    for key in extra_dic:
        if key not in rover_dic:
            rover_dic[key] = extra_dic[key].replace(" ","")
    with DatadirWriter(Path(rover_text).parent) as writer:  
        subwriter = writer[Path(rover_text).stem]
        for key in rover_dic:
            subwriter[key] = rover_dic[key].replace(" ","")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_rootdir",type=Path,default="avsr/exp_far")
    parser.add_argument( "--sub_modeldirs",type=str,nargs="+",default=[])
    parser.add_argument( "--refpath",type=str,default="avsr/data/ground_truth/av_sum_ground_truth.txt")
    parser.add_argument( "--subresultdir",type=str,default="4")
    parser.add_argument( "--evalname",type=str,default="sum_far_farlip")
    parser.add_argument("--wordmode", type=strtobool, default=False)
    parser.add_argument(
        "--distance_type",
        type=str,
        default="cosine",
        help="cosine, euclidean, seuclidean",
    )
    parser.add_argument(
        "--save_align",
        type=strtobool,
        default=False,
        help="save alignment file for analysis",
    )
    parser.add_argument(
        "--cluster_as_session", # In misp2022 we cluster all utterance from the same session and speaker then compare it with ground truth
        type=strtobool,
        default=False,
        help="cluster uttid to session level or no orcle asr",
    )
    parser.add_argument(
        "--extra_file", # In misp2022 chanllenge, some part of evaluation data only have audio(without video), so we both use asr and avsr to get the results, here we add detection result from asr
        type=str,
        default=None,
        help="extra detection result from asr",
    )

    

    args = parser.parse_args()

    resultdir = args.model_rootdir / "rover_result" / args.subresultdir
    print(f"rover subdir:{resultdir}")
    resultdir.mkdir(parents=True, exist_ok=True)
    result_text = resultdir / "rover_text" 
    cer_file = str(resultdir / "cer")
    model_recoder = str(resultdir / "model_recoder")

    model_recoder=open(model_recoder,"w")
    source_list = []
    for sub_modeldir in args.sub_modeldirs:
        print(f"{sub_modeldir}",file=model_recoder)
        try: 
            tokens = list(args.model_rootdir.glob(f"{sub_modeldir}/decode_asr2*5best/*/{args.evalname}/text"))
            if len(tokens) == 0:
                tokens = list(args.model_rootdir.glob(f"{sub_modeldir}/decode_asr2*5best/{args.evalname}/text"))
            source_list.append(tokens[0])
        except:
            print(f"{str(args.model_rootdir / sub_modeldir)} doesn;t exists /decode_asr2*5best/*/{args.evalname}")
            exit(1)
        
    model_recoder.close()


    print("Combin hyps: {} in {}".format(args.sub_modeldirs,args.model_rootdir))
    loss = naive_loss
    rover(source_list,str(result_text),loss,save_align=args.save_align,wordmode=args.wordmode)
    if args.extra_file:
        print(f"add extra decode transcription from {args.extra_file}")
        add_extra_file(str(result_text),args.extra_file)
    rover_token_path = createchar(str(result_text))

    if args.cluster_as_session:
        deal_with_result_decode_and_text(rover_token_path,str(result_text.parent / "session_text.char"))
        deal_with_result_decode_and_text(args.refpath,str(result_text.parent / "session_gt.char"))
        print("Text has been done. You'd better use kaldi tool to compute cer to save your time")
    else:
        result = cer(rover_token_path,args.refpath)
        with open(cer_file,"w") as outputfile:
            print(result)
            print(result,file=outputfile)

    
    
    
