"""SpeMixment module."""
from typing import Optional
from typing import Sequence
from typing import Union
import random
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.mask_along_axis import MaskAlongAxis
from espnet2.layers.mask_along_axis import MaskAlongAxisVariableMaxWidth
from espnet2.layers.time_warp import TimeWarp
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
import logging
import torch
import numpy as np
class SpeMix(AbsSpecAug):
    """Implementation of SpeMix.
    Reference:
        Daniel S. Park et al.
        "SpeMixment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"

    .. warning::
        When using cuda mode, time_warp doesn't have reproducibility
        due to `torch.nn.functional.interpolate`.

    """

    def __init__(
        self,
        mport:float,
        alpha:float,
    ):
       super().__init__()
       self.mport = mport
       self.alpha = alpha
       self.bsize = None
       self.randpairs = []

    def forward(self,*batch):
        assert len(batch) % 2 == 0 , "batch must be like feats+feats_legths"
        batch = list(batch)
        bsize = batch[0].shape[0]
        mixnum = max(bsize * self.mport,1) 
        logging.info(f"bsize:{bsize},mixnum:{mixnum}")
        randpairs = [random.sample(range(bsize), 2) for _ in range(int(mixnum))]
        self.randpairs = randpairs
        self.bsize = bsize
        for pair in randpairs:
            for i in range(0,len(batch),2):
                feats,feats_lengths = batch[i],batch[i+1]
                mixfeat = feats[pair[0]]*self.alpha + feats[pair[1]]*(1-self.alpha)
                batch[i] = torch.cat([batch[i],mixfeat.unsqueeze(0)],0)
                batch[i+1] = torch.cat((batch[i+1],torch.min(feats_lengths[pair[0]],feats_lengths[pair[1]]).unsqueeze(0)),0)
        return batch 

    def fix(self,encoder_out,encoder_out_lens,text,text_lengths):
        tmptext = []
        bsize = self.bsize
        for id,pair in enumerate(self.randpairs):
            insertid = bsize+1+2*id
            if insertid ==  bsize-1+2*len(self.randpairs):
                encoder_out = torch.cat([encoder_out[:insertid],encoder_out[insertid-1].unsqueeze(0)],0)
                encoder_out_lens = torch.cat([encoder_out_lens[:insertid],encoder_out_lens[insertid-1].unsqueeze(0)],0) 
            else:
                encoder_out = torch.cat([encoder_out[:insertid],encoder_out[insertid-1].unsqueeze(0),encoder_out[insertid:]],0)
                encoder_out_lens = torch.cat([encoder_out_lens[:insertid],encoder_out_lens[insertid-1].unsqueeze(0),encoder_out_lens[insertid:]],0)
            text = torch.cat([text,text[pair[0]].unsqueeze(0),text[pair[1]].unsqueeze(0)],0)
            text_lengths = torch.cat([text_lengths,text_lengths[pair[0]].unsqueeze(0),text_lengths[pair[1]].unsqueeze(0)],0)
        return encoder_out,encoder_out_lens,text,text_lengths


    def gen_mixinfo(self):
        idexes = np.array([0]*self.bsize+[1,2]*len(self.randpairs))
        mixinfo = dict(idexes=idexes,alpha=self.alpha,bsize=self.bsize)
        return mixinfo
          

