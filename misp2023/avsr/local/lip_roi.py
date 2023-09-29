from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse
import json
def gen_roiscp(pt_dir,roiscpdir,filename,splitnum=None):
    if splitnum:
        if len(splitnum.split("-"))==1:
            use_splitnums = [int(splitnum)]
        else:
            splitmin,splitmax = splitnum.split("-")
            use_splitnums = list(range(int(splitmin),int(splitmax+1)))
        print(f"selecting splitnum: {use_splitnums}")
    
    print(pt_dir)
    print(roiscpdir)
    files = pt_dir.glob("*.pt")
    with DatadirWriter(roiscpdir) as writer:
        subwriter = writer[filename]
        for file in files:
            if splitnum:
                if len(file.stem.split("@"))-1 in use_splitnums:
                    subwriter[file.stem] = str(file)
            else:
                subwriter[file.stem] = str(file)

def json2scp(srcpath,shapename,tgtdir):
    wavfile = Path(tgtdir) / "wav.scp"
    assert wavfile.exists(),f"there is no {wavfile}"
    wav2path = read_2column_text(wavfile)
    tgtpath = Path(tgtdir) / shapename
    with open(srcpath,"r") as f:
        dic = json.load(f)
    tgtdir = tgtpath.parent
    filename = tgtpath.name
    with DatadirWriter(tgtdir) as writer:
        subwriter = writer[filename]
        for key in wav2path.keys():
            subwriter[key] = str(int(dic[key][0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("lip_roi")
    parser.add_argument( "--pt_dir",type=str,default="pt",
    help="the path where roi.pt files stored",)
    parser.add_argument( "--roiscpdir", type=str, default="dump/raw/org/eval_mid_lip", help="the path where roi.scp storened")
    parser.add_argument( "--filename", type=str, default="roi.scp", help="the path where roi.scp storened")
    parser.add_argument( "--shapename", type=str, default=None, help="key2shape.json to shapename")
    parser.add_argument( "--splitnum", type=str, default=None, help="2-3 mean use spliting 2-3")
    args = parser.parse_args()
    if args.shapename:
        print(f"preparing shape file named {args.shapename} ")
        json2scp(Path(args.pt_dir).parent / "key2shape.json",args.shapename,Path(args.roiscpdir))