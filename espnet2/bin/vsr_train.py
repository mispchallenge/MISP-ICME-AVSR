#!/usr/bin/env python3
from espnet2.tasks.vsr import VSRTask


def get_parser():
    parser = VSRTask.get_parser()
    return parser


def main(cmd=None):
    r"""VSR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    VSRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
