"""SpecAugment module."""
from typing import Optional
from typing import Sequence
from typing import Union
import torch
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.mask_along_axis import MaskAlongAxis,ShiftAlongAxis

class AttAug(AbsSpecAug):
    """Implementation of SpecAug.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"

    .. warning::
        When using cuda mode, time_warp doesn't have reproducibility
        due to `torch.nn.functional.interpolate`.

    """

    def __init__(
        self,
        attentionaug_layernum: int = 3, 
        apply_time_shift: bool = True,
        time_shift_pro: float = 0.2,
        shift_range: int = 10,
        shift_pos: str = None, #"pos","nge",None
        shift_orient: str = "row", #"row,"col"
        apply_time_mask: bool = True,
        time_mask_pro: float = 0.2,
        time_mask_width_range: Optional[Union[int, Sequence[int]]] = 5,
        num_time_mask: int = 2,
        
    ):  
        self.time_shift_pro = time_shift_pro
        self.time_mask_pro = time_mask_pro
        self.attentionaug_layernum = attentionaug_layernum
        if shift_pos not in ["pos","neg",None]:
            raise ValueError(
                f"Either one of pos,neg,None should be applied to shift_pos but got {shift_pos}"
            )
        if shift_orient not in ["row","col"]:
            raise ValueError(
               f"Either one of row,col should be applied to shift_pos but got {shift_orient}"
            )

        if not apply_time_shift and not apply_time_mask:
            raise ValueError(
                "Either one of time_warp, time_mask, or freq_mask should be applied"
            )
        super().__init__()
   
        self.apply_freq_mask = apply_time_shift
        self.apply_time_mask = apply_time_mask

        if apply_time_shift:
            self.time_shift = ShiftAlongAxis(shift_range=shift_range, shift_pos=shift_pos, shift_orient=shift_orient)
        else:
            self.time_shift = None


        if apply_time_mask:
            if shift_orient == "row":
                self.time_mask = MaskAlongAxis(
                    dim="time", #Note 
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                )
            elif shift_orient == "col":
                self.time_mask = MaskAlongAxis(
                    dim="time", #Note 
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                )
        else:
            self.time_mask = None

    def forward(self,embs):
        pro1 = torch.rand(1)
        pro2 = torch.rand(1)
        bsize,fnum,cnum = embs.shape 
        mask = torch.eye(fnum)
        mask = mask.repeat(bsize*self.attentionaug_layernum,1,1)
        if self.time_shift is not None and pro1 < self.time_shift_pro:
            mask = self.time_shift(mask)
        if self.time_mask is not None and pro2 < self.time_mask_pro:
            mask, _ = self.time_mask(mask,None)
        mask.to(embs.device)
        mask = mask.split(bsize)
        return mask


if __name__ == "__main__":
    # attentionaug = AttentionAug(shift_orient="row",shift_pos="neg",attentionaug_layernum=1,shift_range=2,time_mask_pro=0,time_shift_pro=1,apply_time_mask=False)
    attentionaug = AttentionAug(time_mask_width_range=2,time_mask_pro=1,time_shift_pro=1,attentionaug_layernum=1,num_time_mask=1,shift_range=2)
    embs = torch.rand(2,5,512)
    masks = attentionaug(embs)
    print(len(masks))
    print(masks[0].shape)
    print(masks[0])