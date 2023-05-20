import re
import subprocess
from pathlib import Path
import multiprocessing
import argparse
def expdir2cer(expdir,logfile=None):
    if logfile:
        logfile = open(logfile,mode="a")
    expdir = Path(expdir)
    for expdir in expdir.parent.glob(expdir.stem):
        if logfile:
            print("#"*50+expdir.stem+"#"*50,file=logfile)
        print("#"*50+expdir.stem+"#"*50)
        for decodedir in expdir.glob("decode*"):
            for resultpath in  decodedir.rglob("score_cer/result.txt"):
                print_filepath = str(resultpath).replace("/yrfs2/cv1/hangchen2/espnet/mispi/avsr/","")
                with open(resultpath) as f:
                    text = f.read()
                    res = r"\|\s*Sum\s*\|\s*\d*\s*(\d*)\s*\|\s*\d*\s*\d*\s*\d*\s*\d*\s*(\d*)"
                    result = re.findall(res,text,re.S|re.M)
                    cer = float(result[0][1])*100 / float(result[0][0])
                    print("cer:{:.3f}".format(cer))
                    print(print_filepath)
                    if logfile:
                        print("cer:{:.3f}".format(cer),file=logfile)
                        print(print_filepath,file=logfile)
    if logfile:
        logfile.close()
                    
def deal_with_result_decode_and_text(result_decode_path,output_path_from_rd,texttype):
    # make sure ground truth and prediction  are sorted 
        content_result_decode = []
        with open(result_decode_path, 'r') as f_rd:
            for line in f_rd.readlines():
                line = line.strip()
                content_result_decode.append(line)
        output_from_result_decode = []
        for line in content_result_decode:
            if "@" in line :
                file_name = "_".join(line.split(" ")[0].split("@")[0].split("_")[:-1])
            else:
                file_name = "_".join(line.split(" ")[0].split("_")[:-1])
            if texttype == "char":
                text = [char for char in ("").join(line.split(" ")[1:])]
            else: 
                text = line.split(" ")[1:]
            if len(output_from_result_decode) == 0 or output_from_result_decode[-1].split(" ")[0] != file_name:
                output_from_result_decode.append(" ".join([file_name] + text ))
            else:
                output_from_result_decode[-1] += " "+" ".join(text)
        with open(output_path_from_rd, 'w') as f_ord:
            f_ord.write("\n".join(output_from_result_decode)+"\n")

def cluster_cer(gtfile,decode_path,texttype):
        decode_path=Path(decode_path)
        gtfile = Path(gtfile)
        deal_with_result_decode_and_text(str(gtfile),str(decode_path / "cluster_ground_truth"),texttype)
        processes = []
        for text_path in decode_path.rglob("text"):
            if "logdir" not in str(text_path):
                subdirdecode = str(text_path).replace(str(decode_path),"")[1:]
                print(text_path.parent / "cer.log")
                deal_with_result_decode_and_text(str(text_path),str(text_path.parent / "cluster_text"),texttype)
                cmd = "./local/cer.sh --ref {} --hyp {} > {}".format(str(text_path.parent / "cluster_text"),str(decode_path / "cluster_ground_truth"),str(text_path.parent / "cer.log"))
                processes.append(subprocess.Popen(cmd, shell=True))
        for process in processes:
            process.wait()
        print("CER Done!") 
        cluster_log = decode_path / "cluster_cer.log"
        if cluster_log.exists():
            cluster_log.unlink()
        for text_path in sorted(list(decode_path.rglob("text"))):
            if "logdir" not in str(text_path):
                subdirdecode = str(text_path).replace(str(decode_path),"")[1:]
                cerfile = str(text_path.parent / "cer.log")
                cmd = "echo \"{}\" >> {} && cat {} >> {}  ".format(subdirdecode, str(cluster_log), cerfile, str(cluster_log))
                subprocess.run(cmd, shell=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("cer_tools")
    parser.add_argument( "--mode", type=str, default=None, help="cluster_cer,cer,result_cer")

    ##cluster_cer
    parser.add_argument( "--gtfile", type=str, default=None, help="")
    parser.add_argument( "--decode_path", type=str, default=None, help="")
    parser.add_argument( "--texttype", type=str, default="char", help="")
    args = parser.parse_args()
    if args.mode == "cluster_cer":
        cluster_cer(args.gtfile,args.decode_path,args.texttype)

