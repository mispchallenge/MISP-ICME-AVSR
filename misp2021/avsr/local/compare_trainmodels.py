import torch 
from pathlib import Path
import argparse
from espnet2.train.reporter import Reporter
from collections import defaultdict
from numpy import *
import numpy as np

def load_criterion(checkpoint_paths,nbest,criterions):
    rdict = defaultdict(list)
    cutepochdict = defaultdict(list)
    for checkpoint_path in checkpoint_paths:
        states = torch.load(checkpoint_path)
        reporter = Reporter()
        reporter.load_state_dict(states["reporter"])
        for ph, k, m in criterions:
            c_seg = ("_").join([ph,k,m])
            rlist = reporter.sort_epochs_and_values(ph, k, m)[: nbest]
            clist = [ epoch_value[1] for epoch_value in rlist ]
            rdict[c_seg].append(rlist)
            cutepochdict[c_seg].append(clist)
    return rdict,cutepochdict

def sort_dict(rdict,cutepochdict,model_dirs,reduce=None,show_mode=None):
    reduct_dict = {"max":max,"mean":mean,"min":min}
    for c_seg in rdict:
        reduce = reduce if reduce else c_seg.split("_")[-1]
        reduce_list = [ reduct_dict[reduce](vlist) for  vlist in cutepochdict[c_seg] ]
        indexes = np.argsort(reduce_list)
        if c_seg.split("_")[-1] == "max":
            indexes = indexes[::-1]
        print(f"{c_seg}:")
        for index in indexes:
            if show_mode == "details":
                print(f"{model_dirs[index]}:{reduce_list[index]}   {rdict[c_seg][index]}")
            else: 
                print(f"{model_dirs[index]}:{reduce_list[index]}")

        



if __name__ == '__main__':
    #compare different models according input criterion
    parser = argparse.ArgumentParser("run_wpe")
    parser.add_argument( "--root_dir",type=str,default="avsr/exp_far")
    parser.add_argument( "--model_dirs",type=str,nargs="+",default=["AVnewcross_ivsr_lipfmid_triphone01","AVnewcross_ivsr_lipfmid_triphone11","AVnewcross_ivsr_lipfmid_triphone21","AVnewcross_ivsr_lipfmid_triphone31"])
    parser.add_argument( "--criterions",type=str,nargs='+',default=["valid acc max"])
    parser.add_argument( "--nbest",type=int,default=3)
    parser.add_argument( "--reduce",type=str,default=None)
    parser.add_argument( "--show_mode",type=str,default=None)
    args = parser.parse_args()
    assert args.reduce in ["mean",None]

    model_dirs = args.model_dirs
    criterions = [tuple(crit.split(" ")) for crit in args.criterions]
    root_dir = Path(args.root_dir)
    checkpoint_paths = []
    match_paths = []
    for globdir in model_dirs:
        matchdirs = list(root_dir.glob(globdir))
        if len(matchdirs) == 0:
            print(f"there isn't any path match the glob path of {str(root_dir / globdir)}" )
        else:
            checkpoint_paths.extend([root_dir / matchdir / "checkpoint.pth" for matchdir in matchdirs])
            match_paths.extend([matchdir for matchdir in matchdirs])
    rdict,cutepochdict = load_criterion(checkpoint_paths,args.nbest,criterions)
    sort_dict(rdict,cutepochdict,match_paths,args.reduce,args.show_mode)
    