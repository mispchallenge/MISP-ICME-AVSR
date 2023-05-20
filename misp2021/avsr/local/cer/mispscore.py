#!/usr/bin/python
# -*- coding:utf-8 -*- 
import numpy as np
import codecs
import os
import sys
import glob
##strong function by checking space row,ending with \r or \n
def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r',encoding='utf-8') as handle:
            lines_content = handle.readlines()
            processed_lines = []
            for line_content in lines_content:
                line_content = line_content.replace('\n', '').replace('\r', '')
                if line_content.strip():
                     processed_lines.append(line_content)
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n','\r'] else '{}\n'.format(x), lines_content)]
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None


def compute_cer_web(src_seq,tgt_seq):
    src_seq = src_seq.replace("<UNK>","*")
    hypothesis = list(src_seq)
    reference = list(tgt_seq)
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1
                compare_val = [substitution, insertion, deletion]   # 优先级

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1
            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        # 出边界处理
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    return nb_map["N"],nb_map["I"],nb_map["D"],nb_map["S"] 

def input_interface(src_file,tgt_file):
    src_lines = sorted(text2lines(textpath=src_file))
    src_dic = {}
    for src_line in src_lines:
        key,*value = src_line.split(' ')
        src_dic[key] = value

    tgt_lines = sorted(text2lines(textpath=tgt_file))
    tgt_dic = {}
    for tgt_line in tgt_lines:
        key,*value = tgt_line.split(' ')
        tgt_dic[key] = value

    result_dic = {}
    sum_entire = 0
    sum_insert = 0
    sum_delete = 0
    sum_substitute = 0
    

    #check input keys, it must include reference keys 
    if not sorted(list(tgt_dic.keys()))==sorted(list(set(tgt_dic.keys()) & set((src_dic.keys())))):
        watch_out = "In your submission, some segments are missing. Please check and submit again!"
    else: watch_out = ""

    miss_utter = []
    for key,tgt_list in list(tgt_dic.items()):
        if key in src_dic:
            src_list =  src_dic.get(key)
        else:
            src_list=['']
            miss_utter.append(key)
        entire,insert,delete,substitute = compute_cer_web(src_seq=''.join(src_list),tgt_seq=''.join(tgt_list))
        sum_entire += entire
        sum_insert += insert
        sum_delete += delete
        sum_substitute += substitute
        result_dic[key] ={'entire':entire,'insert':insert,'delete':delete,'substitute':substitute}
    return [(sum_insert+sum_delete+sum_substitute)/sum_entire,sum_entire,sum_insert,sum_delete,sum_substitute],watch_out,miss_utter

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument( "--src_file",type=str,default="",help="dump/raw/eval_far")
    parser.add_argument( "--tgt_file",type=str,default="",help="shapefilepath",)
    args = parser.parse_args()
    result,watch_out,miss_utter = input_interface(src_file=args.src_file,tgt_file=args.tgt_file)
    print("CCER: {:.2f}".format(result[0]*100)) 

       
