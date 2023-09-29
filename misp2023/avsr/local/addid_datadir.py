from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
from pathlib import Path 
import argparse
from espnet2.fileio.read_text import read_2column_text
if __name__ == '__main__':
    #add self-define ids to each utterance uid's to different the utterance which has the same uid but from different datadir before mixup then to a bigger datadir
    # e.g. uid(near olddatadir) + uid(far olddatadir) -> near_uid+far_uid (newdatadir)
    parser = argparse.ArgumentParser('')
    parser.add_argument('--addids', nargs='+',type=str, default='') #note "kong" means add nothing 
    parser.add_argument('--movefiles', nargs='+',type=str, default='')
    parser.add_argument('--tgtdir',type=str, default='')
    parser.add_argument('--srcdirs',nargs='+',type=str, default='')
    parser.add_argument('--cutsp',type=str, default=False) #if uid include "sp" then drop it; we don't use speed pertubation in avsr training
    parser.add_argument('--channelmode',nargs="+",type=bool, default=[]) #channel utterance is storen as [channel,time],so for each channel’uid we add channel_num as the fix to different each channel
    parser.add_argument('--tgtsubdirs',nargs="+",type=str, default=None)
    parser.add_argument('--sufix',type=bool, default=False)
    args = parser.parse_args()

    srcdirs = args.srcdirs
    tgtdir = Path(args.tgtdir)
    addids = args.addids
    addids = ["" if id=="kong" else id  for id in addids  ] 
    movefiles = args.movefiles
    channelmode = args.channelmode
    channelnumdic = {"far":6,"mid":2}
    assert len(addids)==len(srcdirs)
    if channelmode:
        len(addids)==len(srcdirs)==len(channelmode)
    else:
        channelmode = [False]*len(addids)
    
    tgtdir.mkdir(exist_ok=True)
    for id,srcdir in enumerate(srcdirs):
        srcdir = Path(srcdir) 
        if args.tgtsubdirs:
            tgtdir0 = tgtdir / args.tgtsubdirs[id]
        else:
            if args.cutsp:
                tgtdir0 = tgtdir / srcdir.name.replace("_sp","")
            else:
                tgtdir0 = tgtdir / srcdir.name
        with DatadirWriter(tgtdir0) as writer:
            for file in movefiles:
                subwriter = writer[file]
                filedic = read_2column_text(str(srcdir / file))
                if channelmode[id]:
                    channel_pos = addids[id].replace("channel","")
                    channel_num = channelnumdic.get(channel_pos)
                    assert channel_num, "you should use channelfar or channelmid as channel file addid"
                    for key,vaule in filedic.items():
                        for i in range(channel_num):
                            if addids[id]:
                                if args.sufix:
                                    subwriter[key+"_"+str(i)+"_"+addids[id]] = vaule
                                else:
                                    subwriter[addids[id]+"_"+key+"_"+str(i)] = vaule
                            else:
                                subwriter[key+"_"+str(i)] = vaule
                else:
                    for key,vaule in filedic.items():
                        if args.cutsp and "sp" in str(srcdir) and "sp" in key:
                            continue
                        if addids[id]:
                            if args.sufix:
                                subwriter[key+"_"+addids[id]] = vaule
                            else:
                                subwriter[addids[id]+"_"+key] = vaule
                        else:
                            subwriter[key] = vaule
                

    


