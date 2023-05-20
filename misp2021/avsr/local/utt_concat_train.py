from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse
import json
from collections import defaultdict

def json2dic(jsonpath, dic=None, format=False):
    """
    read dic from json or write dic to json
    :param jsonpath: filepath of json
    :param dic: content dic or None, None means read
    :param format: if True, write formated json, it needs more time
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(jsonpath, 'r') as handle:
            output = json.load(handle)
        return output
    else:
        assert isinstance(dic, dict)
        with codecs.open(jsonpath, 'w') as handle:
            if format:
                json.dump(dic, handle, indent=4, ensure_ascii=False)
            else:
                json.dump(dic, handle)
        return None
    
#an extensive version to pdfdir
def gen_pdfconcatdir(datadir,catdir,max_sample,concatnum,step,pdffilename):
    
    id2wav = read_2column_text(datadir / "wav.scp")
    id2pdf =  read_2column_text(datadir / pdffilename)
    id2wavshape =  read_2column_text(datadir / "speech_shape")
    id2pdfshape =  read_2column_text(datadir / f"pdf_shape")
    session_uttids = defaultdict(list)
    for id in id2wav.keys():
        if "sp" not in id:
            sid = ("_").join(id.strip().split("_")[:-1])
            session_uttids[sid].append(id)
    with DatadirWriter(catdir) as writer:
        utt2wav = writer["wav.scp"]
        utt2wavshape = writer["speech_shape"]
        utt2uniq = writer["utt2uniq"]
        utt2spk = writer["utt2spk"]
        utt2pdf = writer[pdffilename]
        utt2pdfshape = writer[f"pdf_shape"]
    
        for sid in session_uttids:
            for i in range(0,len(session_uttids[sid]),step):
                if i+concatnum-1 >= len(session_uttids[sid]):
                    break
                utterlist = session_uttids[sid][i:i+concatnum]
                catid = "@".join(utterlist)
                #check duration
                duration = sum([int(id2wavshape[id]) for id in utterlist])
                if max_sample:
                    if duration >= max_sample:
                        continue
                utt2wav[catid] = "@".join([id2wav[id] for id in utterlist])
                utt2wavshape[catid] = str(duration)
                utt2uniq[catid] = catid 
                utt2spk[catid] = sid.split("_")[0]
                utt2pdf[catid] = "@".join([id2pdf[id] for id in utterlist])
                utt2pdfshape[catid] = str(sum([int(id2pdfshape[id]) for id in utterlist]))

        
#give a data directory to return a cat datadir 
# 1.wav.scp,text,text_shape,wav_shape -> wav.scp, utt2uniq,utt2spk,text,text_shape,wav_shape
def gen_concatdir(datadir,catdir,max_sample,concatnum,step,toke_type="char"):
    
    id2wav = read_2column_text(datadir / "wav.scp")
    id2text =  read_2column_text(datadir / "text")
    id2wavshape =  read_2column_text(datadir / "speech_shape")
    id2textshape =  read_2column_text(datadir / f"text_shape.{toke_type}")
    session_uttids = defaultdict(list)
    for id in id2wav.keys():
        if "sp" not in id:
            sid = ("_").join(id.strip().split("_")[:-1])
            session_uttids[sid].append(id)
    with DatadirWriter(catdir) as writer:
        utt2wav = writer["wav.scp"]
        utt2uniq = writer["utt2uniq"]
        utt2spk = writer["utt2spk"]
        utt2text = writer["text"]
        utt2textshape = writer[f"text_shape.{toke_type}"]
        utt2wavshape = writer["speech_shape"]

        for sid in session_uttids:
            for i in range(0,len(session_uttids[sid]),step):
                if i+concatnum-1 >= len(session_uttids[sid]):
                    break
                utterlist = session_uttids[sid][i:i+concatnum]
                catid = "@".join(utterlist)
                #check duration
                duration = sum([int(id2wavshape[id]) for id in utterlist])
                if max_sample:
                    if duration >= max_sample:
                        continue
                utt2wav[catid] = "@".join([id2wav[id] for id in utterlist])
                utt2wavshape[catid] = str(duration)
                utt2uniq[catid] = catid 
                utt2spk[catid] = sid.split("_")[0]
                utt2text[catid] = "".join([id2text[id] for id in utterlist])
                try:
                    utt2textshape[catid] = str(sum([int(id2textshape[id].split(",")[0]) for id in utterlist])) + "," + id2textshape[utterlist[0]].split(",")[1]     
                except:
                    utt2textshape[catid] = str(sum([int(id2textshape[id]) for id in utterlist]))
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument( "--datadir",type=str,default="avsrdump/raw/org/gss_train_far_spcat1_5")
    parser.add_argument( "--catdatadir", type=str, default="avsrdata/gss_train_far/text", help="the path where roi.scp storened")
    parser.add_argument( "--maxduration", default=None, type=float)
    parser.add_argument( "--concatnum", default=2, type=int)
    parser.add_argument( "--step", default=1,type=int)
    parser.add_argument( "--fps", default=16000,type=int)
    parser.add_argument( "--toke_type", default="char",type=str)
    parser.add_argument( "--mode", default=None,type=str)
    parser.add_argument( "--pdffilename", default=None,type=str)
    
    args = parser.parse_args()
    max_sample = round(args.fps*args.maxduration) if args.maxduration else None
    datadir = Path(args.datadir)
    catdir = Path(args.catdatadir)
    if args.mode=="pdf":
        gen_pdfconcatdir(datadir,catdir,max_sample,args.concatnum,args.step,args.pdffilename)
    else:
        gen_concatdir(datadir,catdir,max_sample,args.concatnum,args.step,args.toke_type)


