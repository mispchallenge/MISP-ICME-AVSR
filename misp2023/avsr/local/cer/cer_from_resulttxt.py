import re

from pathlib import Path
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
                with open(resultpath) as f:
                    text = f.read()
                    res = r"\|\s*Sum\s*\|\s*\d*\s*(\d*)\s*\|\s*\d*\s*\d*\s*\d*\s*\d*\s*(\d*)"
                    result = re.findall(res,text,re.S|re.M)
                    cer = float(result[0][1])*100 / float(result[0][0])
                    print("cer:{:.3f}".format(cer))
                    print(resultpath)
                    if logfile:
                        print("cer:{:.3f}".format(cer),file=logfile)
                        print(resultpath,file=logfile)
    if logfile:
        logfile.close()
                    

if __name__ == "__main__":
    expdirs = ["exp_far/AVffn_ivsr_lipfmid_lrw",
                "exp_far/AVffn_ivsr_lipfmid_lrw1000_olda",]
    logfile = ""
    for expdir in expdirs:
        expdir2cer(expdir,logfile)
        


