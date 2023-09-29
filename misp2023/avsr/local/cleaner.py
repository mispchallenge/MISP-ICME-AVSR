import shutil
import os
from pathlib import Path
def rmsub(dirpath,subdirlist,drop_model_mode=None):
    dirpath = Path(dirpath)
    for subdir in subdirlist:
        subdirs = dirpath.glob(subdir)
        for subdir in subdirs:
            print(subdir)
            if subdir.exists():            
                shutil.rmtree(str(subdir))
    if drop_model_mode == "unimp":
        models = list(dirpath.glob("*.pth"))
        save_models =  [str(model) for model in models if model.is_symlink()]
        save_models.extend([str(dirpath / str(os.readlink(model))) for model in save_models]) 
        save_models += [str(dirpath / "checkpoint.pth")]
        for model in models:
            if str(model) not in save_models:
                os.remove(str(model))
    elif drop_model_mode == "all":
        models = dirpath.glob("*.pth")
        for model in models:
            os.remove(str(model))
            


if __name__ == "__main__":
    # Del some unimportant subdir (like history cache model, attention visualization img, and decoder logfiles)in expdirs
    rootdir = Path("exp_vsr/.clean")
    subdirlist = ["att_ws","tensorboard","decode*/*/score_*","decode*/*/logdir","decode*/*/*/score_*","decode*/*/*/logdir"]
    print(f"cleaning {rootdir}")
    for dir in rootdir.iterdir():
        rmsub(dir,subdirlist,drop_model_mode="unimp")
