# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional
from typing import Tuple
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
import logging
import torch
from espnet2.asr.encoder.utils import DimConvert,NewDimConvert
from typeguard import check_argument_types
from torch import nn
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer,CrossAttentionFusionEncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
    TraditionMultiheadRelativeAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.encoder.abs_encoder import AbsEncoder,AVOutEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from .network_audio_visual_fusion import AudioVisualFuse

class AttentionKQVFusion(nn.Module):
    """CrosschannelLayer layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """
    def __init__(
        self,
        size,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        input_mask:bool=True,
        output_mask:bool=True,
    ):
        """Construct an channelwiseLayer object."""
        super(AttentionKQVFusion, self).__init__()
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size) #for k
        self.norm2 = LayerNorm(size) #for q,v
        self.norm3 = LayerNorm(size) 
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.input_mask=input_mask
        self.output_mask=output_mask

    def forward(self, x_q, x_kv, mask):
        """Compute encoded features.
        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if not self.input_mask:
            mask = (~make_pad_mask(mask)[:, None, :]).to(x_q.device) 
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.

        #pos emb
        pos_emb = None
        if isinstance(x_kv, tuple):
            x_kv, pos_emb = x_kv[0], x_kv[1]
        if isinstance(x_q, tuple):
            x_q = x_q[0]
            
        #MHA 
        residual = x_kv
        if self.normalize_before:
            x_q = self.norm1(x_q)
            x_kv = self.norm2(x_kv)
        if pos_emb != None :
            x = residual + self.dropout(
                    self.src_attn(x_q, x_kv, x_kv, pos_emb, mask)
                )
        else:
            x = residual + self.dropout(
                    self.src_attn(x_q, x_kv, x_kv, mask)
                )
        if not self.normalize_before:
            x = self.norm1(x)
    
        #forward layer
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        
        if not self.output_mask:
            mask = mask.squeeze(1).sum(1)
        
        if pos_emb is not None:
            return (x, pos_emb), mask
            
        return x, mask

class AVConformerEncoder(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        feat_dim:int,
        alayer_num1:int,
        alayer_num2:int,
        alayer_num3:int,
        vlayer_num1:int,
        vlayer_num2:int,
        vlayer_num3:int,
        avlayer_num:int
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.input_layer = Conv2dSubsampling(
                feat_dim,
                conformer_conf["output_size"],
                conformer_conf["dropout_rate"],
                None)

        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf) #incluee embedding layer
        self.vlayer2 = ConformerEncoder(num_blocks=vlayer_num2,**conformer_conf)
        self.vlayer3 = ConformerEncoder(num_blocks=vlayer_num3,**conformer_conf)
        self.fusion = torch.nn.Sequential(
                        torch.nn.Linear(conformer_conf["output_size"]*2, conformer_conf["output_size"]),
                        torch.nn.LayerNorm(conformer_conf["output_size"]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(conformer_conf["dropout_rate"]),              
        )  
        self.avlayer = ConformerEncoder(num_blocks=avlayer_num,**conformer_conf)
          
         
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """

        # aduio feat and video subsampling
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        org_feats,org_masks = self.input_layer(feats,masks)
        org_feats_lengths = org_masks.squeeze(1).sum(1)
        masks = (~make_pad_mask(video_lengths)[:, None, :]).to(video.device)
       
       
        #layer 1
        outfeats1,outfeats_lengths1,_ = self.alayer1(org_feats,org_feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,org_feats_lengths)

        #fusion 1+layer 2
        x_concat = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1= self.fusion(x_concat)
       
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)
        outvideo2,outvideo_lengths2,_ = self.vlayer2(outvideo1,outvideo_lengths1)

        #skip connection + layer 3
        outfeats3,outfeats_lengths3,_ = self.alayer3(org_feats+outfeats2,outfeats_lengths2)
        outvideo3,outvideo_lengths3,_ = self.vlayer3(video+outvideo2,outvideo_lengths2)

        #fusion 2 + layer 4
        x_concat = torch.cat((outfeats3, outvideo3), dim=-1)
        amid_feat2= self.fusion(x_concat)
        hidden_feat,hidden_feat_lengths,_ = self.avlayer(amid_feat2,outfeats_lengths3)

        return hidden_feat,hidden_feat_lengths,_
              
class AVConformerEncoder2(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        feat_dim:int,
        alayer_num1:int,
        alayer_num2:int,
        alayer_num3:int,
        vlayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.input_layer = Conv2dSubsampling(
                feat_dim,
                conformer_conf["output_size"],
                conformer_conf["dropout_rate"],
                None)
        
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf) #incluee embedding layer
        self.fusion = torch.nn.Sequential(
                        torch.nn.Linear(conformer_conf["output_size"]*2, conformer_conf["output_size"]),
                        torch.nn.LayerNorm(conformer_conf["output_size"]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(conformer_conf["dropout_rate"]),              
        )  
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """

        # aduio downsampling while video has subsampling in frontend 
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        org_feats,org_masks = self.input_layer(feats,masks)
        org_feats_lengths = org_masks.squeeze(1).sum(1)
        
        #fix length
        if not org_feats_lengths.equal(video_lengths):
            org_feats_lengths = org_feats_lengths.min(video_lengths)
            video_lengths = org_feats_lengths.clone()
            feats = feats[:,:max(org_feats_lengths)]
            video = video[:,:max(video_lengths)]

        #fusion 1 + layer 1
        x_concat = torch.cat((org_feats, video), dim=-1)
        amid_feat1= self.fusion(x_concat)
        outfeats1,outfeats_lengths1,_ = self.alayer1(amid_feat1,org_feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,org_feats_lengths)

        #fusion 2+layer 2
        x_concat = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1= self.fusion(x_concat)
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)

        #skip connection + layer av
        hidden_feat,hidden_feat_lengths,_ = self.alayer3(org_feats+outfeats2,outfeats_lengths2)

        return hidden_feat,hidden_feat_lengths,_
      
class AVConformerEncoder3(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        feat_dim:int,
        alayer_num1:int,
        alayer_num2:int,
        vlayer_num1:int,
   
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.input_layer = Conv2dSubsampling(
                feat_dim,
                conformer_conf["output_size"],
                conformer_conf["dropout_rate"],
                None)
        
      
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) 
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf)  
        self.fusion = torch.nn.Sequential(
                        torch.nn.Linear(conformer_conf["output_size"]*2, conformer_conf["output_size"]),
                        torch.nn.LayerNorm(conformer_conf["output_size"]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(conformer_conf["dropout_rate"]),              
        )  
  
          
         
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """

        # aduio feat and video subsampling
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        org_feats,org_masks = self.input_layer(feats,masks)
        org_feats_lengths = org_masks.squeeze(1).sum(1)
        masks = (~make_pad_mask(video_lengths)[:, None, :]).to(video.device)

       
        #fusion 1+layer 1
        x_concat = torch.cat((org_feats, video), dim=-1)
        amid_feat1= self.fusion(x_concat)
        outfeats1,outfeats_lengths1,_ = self.alayer1(org_feats,org_feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,org_feats_lengths)

        #fusion 2+layer 2
        x_concat = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat2= self.fusion(x_concat)
        hidden_feat,hidden_feat_lengths,_ = self.alayer2(amid_feat2,outfeats_lengths1)

        return hidden_feat,hidden_feat_lengths,_

"""
AVConformerEncoder4 is downsampling wav 100fps to 25ps and simply concat them and put into 12 conformer layers 
"""
class AVConformerEncoder4(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.fusion = DimConvert(in_channels=512*2,out_channels=256) 
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
        self.subsampling = Conv2dSubsampling(
                            512,
                            512,
                            conformer_conf["dropout_rate"],
                            None)
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats_lengths.device)
        feats, masks = self.subsampling(feats, masks)
        feats_lengths = masks.squeeze(1).sum(1)
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        #fusion 1+layer 1
        x_concat = torch.cat((feats, video), dim=-1) #B,T,1024
        amid_feat1= self.fusion(x_concat) #B,T,256
        hidden_feat,hidden_feat_lengths,_ = self.alayer1(amid_feat1,feats_lengths)
        return hidden_feat,hidden_feat_lengths,_

"""
Note this is the best model I have test.
AVConformerEncoder5 is similar to AVConformerEncoder2 and is used for wav preencode 25ps +video 25 ps ,which have the same fps;for feat preencode 100ps + video 25ps you can use  AVConformerEncoder6
"""
class AVConformerEncoder5(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int=3,
        alayer_num2:int=3,
        alayer_num3:int=3,
        vlayer_num1:int=3,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf = conformer_conf
       
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf)
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        video_conformer_conf = conformer_conf.copy()
        video_conformer_conf["input_size"] = 512
        video_conformer_conf["input_layer"] = "linear"
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**video_conformer_conf) 
        self.fusion1 = DimConvert(in_channels=512*2,out_channels=256)
        self.fusion2 = DimConvert(in_channels=256*2,out_channels=256)
        self.audioturner = DimConvert(in_channels=256*2,out_channels=256)
      
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, T, 256).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, T, 256). #video and audio both 25 ps
                video_lengths (torch.Tensor): Input length (#batch)
        """
        # both have nearly same dimensional
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]

        #fusion 1 + layer 1
        x_concat = torch.cat((feats, video), dim=-1)
        amid_feat = self.fusion1(x_concat)
        
        # print(f"famid_feat.shape:{amid_feat.shape},outfeats_lengths:{feats_lengths}",file=log_file)
        outfeats1,outfeats_lengths1,_ = self.alayer1(amid_feat,feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,video_lengths)

        #fusion 2+layer 2
        x_concat1 = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1 = self.fusion2(x_concat1)
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)

        #skip connection + layer av
        res = self.audioturner(feats)
        hidden_feat,hidden_feat_lengths,_ = self.alayer3(outfeats2+res,outfeats_lengths2)

        return hidden_feat,hidden_feat_lengths,_
    # conformer_conf = {"output_size": 256  ,  # dimension of attention
    #     "attention_heads": 4,
    #     "linear_units": 2048 , # the number of units of position-wise feed forward
    #     "dropout_rate": 0.1,
    #     "positional_dropout_rate": 0.1,
    #     "attention_dropout_rate": 0.0,
    #     "input_layer": "conv2d" ,# encoder architecture type
    #     "normalize_before": True,
    #     "pos_enc_layer_type": "rel_pos",
    #     "selfattention_layer_type": "rel_selfattn",
    #     "activation_type": "swish",
    #     "macaron_style": True,
    #     "use_cnn_module": True,
    #     "cnn_module_kernel": 15}

    # encoder = AVConformerEncoder5(conformer_conf)
    # feats = torch.rand(16,90,512)
    # video = torch.rand(16,90,512)
    # feats_l = torch.randint(40,91,(16,))
    # feats_l[0] = 90
    # video_l = torch.randint(90,91,(16,))
    # video_l[0] = 90
    # hidden_feat,hidden_feat_lengths,_ = encoder(feats,feats_l,video,video_l)
    # print(hidden_feat.shape,hidden_feat_lengths.shape)

"""
AVConformerEncoder6 is similar to AVConformerEncoder2, and is used for feat preencode 100ps  video 25ps ,it will downsampling feat preencode first
"""
class AVConformerEncoder6(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int=3,
        alayer_num2:int=3,
        alayer_num3:int=3,
        vlayer_num1:int=3,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"] #512
        self.conformer_conf =conformer_conf
       
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf)
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        # self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        video_conformer_conf = conformer_conf.copy()
        video_conformer_conf["input_size"] = 512
        video_conformer_conf["input_layer"] = "linear"
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**video_conformer_conf) 
        self.fusion1 = DimConvert(in_channels=512*2,out_channels=256)
        self.fusion2 = DimConvert(in_channels=256*2,out_channels=256)
        # self.audioturner = DimConvert(in_channels=256*2,out_channels=256)
        self.subsampling = Conv2dSubsampling(
                            80,
                            512,
                            conformer_conf["dropout_rate"],
                            None)
         
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, T, 256).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, T, 256). #video and audio both 25 ps
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #downsampling audio 100fps -> 25fps
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(video_lengths.device)
        feats, masks = self.subsampling(feats, masks)
        feats_lengths = masks.squeeze(1).sum(1)
        # both have nearly same dimensional
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
       
        #fusion 1 + layer 1
        x_concat = torch.cat((feats, video), dim=-1)
        amid_feat = self.fusion1(x_concat)
        
        # print(f"famid_feat.shape:{amid_feat.shape},outfeats_lengths:{feats_lengths}",file=log_file)
        outfeats1,outfeats_lengths1,_ = self.alayer1(amid_feat,feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,video_lengths)

        #fusion 2+layer 2
        x_concat1 = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1 = self.fusion2(x_concat1)
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)
        return outfeats2,outfeats_lengths2,_
        #skip connection + layer av
        # res = self.audioturner(feats)
        # hidden_feat,hidden_feat_lengths,_ = self.alayer3(outfeats2+res,outfeats_lengths2)

        # return hidden_feat,hidden_feat_lengths,_

"""
AVConformerEncoder7 is similar to AVConformerEncoder4, and is used for wav preencode 25ps +video 25 ps ,which don't need  downsampling 
"""
class AVConformerEncoder7(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.fusion = DimConvert(in_channels=512*2,out_channels=256) 
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). 
                video_lengths (torch.Tensor): Input length (#batch)
        """

        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        #fusion 1+layer 1
        x_concat = torch.cat((feats, video), dim=-1) #B,T,1024
        amid_feat1= self.fusion(x_concat) #B,T,256
        hidden_feat,hidden_feat_lengths,_ = self.alayer1(amid_feat1,feats_lengths)
        return hidden_feat,hidden_feat_lengths,_

"""
AVFineTuneConformerEncoder is used for feat vggpreencode 25ps +video 25 ps ,different with other encoder it use audio feat as K, and speech_feat as K,Q, and it will be fine-tune based on a 12-layer asr comferformer
"""
class AVFineTuneConformerEncoder(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        encoder_conf:dict,
        alayer_num1:int=3,
        vlayer_num1:int=3,
        alayer_num2:int=9,
    ):  
        super().__init__()
        #setting configs
        self._output_size = encoder_conf["hidden_size"]
        conformer_conf = encoder_conf["conformer_conf"]
        embed_conf = encoder_conf["embed_conf"]
        self.conformer_conf = conformer_conf
        self.normalize_before = conformer_conf["normalize_before"]
        attentionfusion_conf = encoder_conf["attentionfusion_conf"]
        pos_enc_layer_type = embed_conf["pos_enc_layer_type"]
        selfattention_layer_type = conformer_conf["selfattention_layer_type"]

        #position embedding layer
        pos_enc_class = AVFineTuneConformerEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)
        self.embed = torch.nn.Sequential(
                pos_enc_class(embed_conf["output_size"], embed_conf["positional_dropout_rate"])
            )
        
        #transformer cross_attention v as q, a as k,v 
        fusion_attentionlayer = AVFineTuneConformerEncoder.get_posemb_attention(attentionfusion_conf["MHA_type"],attentionfusion_conf["MHA_conf"],encoder_conf["hidden_size"] )
        positionwise_layer, positionwise_layer_args = AVFineTuneConformerEncoder.get_positionwise_layer(**attentionfusion_conf["positionwise_layer_args"],attention_dim=encoder_conf["hidden_size"])
        self.fusionblock1 = AttentionKQVFusion( size=encoder_conf["hidden_size"],             
                                                src_attn=fusion_attentionlayer,
                                                feed_forward=positionwise_layer(*positionwise_layer_args),
                                                dropout_rate=attentionfusion_conf["dropout_rate"],
                                                normalize_before=attentionfusion_conf["normalize_before"],
                                                input_mask=True,           
                                                output_mask=True)  
        # a & v conformer block
        
        self.alayerblock1 = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=alayer_num1,**conformer_conf) # a 3-layer conformer
        if vlayer_num1 != 0 :
            self.vlayerblock1 = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=vlayer_num1,**conformer_conf) #a 3-layer conformer
        else:
            self.vlayerblock1 = None
            
        #transformer cross_attention v as q, a as k,v 
        if vlayer_num1 != 0:
            self.fusionblock2 = AttentionKQVFusion( size=encoder_conf["hidden_size"],
                                                src_attn=fusion_attentionlayer,
                                                feed_forward=positionwise_layer(*positionwise_layer_args),
                                                dropout_rate=attentionfusion_conf["dropout_rate"],
                                                normalize_before=attentionfusion_conf["normalize_before"],
                                                input_mask=True,           
                                                output_mask=True ) 
        else:
            self.fusionblock2 = None
        self.alayerblock2 = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=alayer_num2,**conformer_conf) #a 9-layer conformer

        if self.normalize_before:
            self.after_norm = LayerNorm(self._output_size)
    @staticmethod
    def get_comformerblocks(
        hidden_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,):
        assert check_argument_types()
        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                hidden_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                hidden_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                hidden_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                hidden_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                hidden_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                hidden_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (hidden_size, cnn_module_kernel, activation)

        return repeat(num_blocks,
                        lambda lnum: EncoderLayer(
                        hidden_size,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                        convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                        dropout_rate,
                        normalize_before,
                        concat_after,
                        ))
    
    @staticmethod
    def get_posembclass(pos_enc_layer_type,selfattention_layer_type):
        #position embedding layer
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        return pos_enc_class

    @staticmethod
    def get_posemb_attention(MHA_type,MHA_conf,size):
        if MHA_type == "selfattn":
            attention_layer = MultiHeadedAttention(**MHA_conf,n_feat=size)
        elif MHA_type == "rel_selfattn":
            attention_layer = RelPositionMultiHeadedAttention(**MHA_conf,n_feat=size)
        elif MHA_type == "trad_selfattn":
            attention_layer = TraditionMultiheadRelativeAttention(
                num_heads=MHA_conf["n_head"],
                embed_dim=size,
                dropout=MHA_conf["dropout_rate"]
                )
        else: 
            logging.error(f"MHA_type must in abs_pos,rel_pos,trad_rel_pos,but got {MHA_type}")
        return attention_layer

    def output_size(self) -> int:
        return self._output_size

    @staticmethod
    def get_positionwise_layer(
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). 
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #modal alignment
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        mask = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        #embd
        feats = self.embed(feats)
        if not isinstance(feats,tuple):
            video = self.embed(video) #add abs emb
        else:
            video = (video,feats[1])
        #fusionblock1
        avfeats,avfeats_mask = self.fusionblock1(x_q=video, x_kv=feats, mask=mask) 
        #alayerblock1
        avfeats,avfeats_mask = self.alayerblock1(avfeats,avfeats_mask)
        #vlayerblock
        if self.vlayerblock1:
            video,video_mask = self.vlayerblock1(video,mask)
        #fusionblock2
        if self.vlayerblock1:
            avfeats,avfeats_mask = self.fusionblock2(x_q=video,x_kv=avfeats,mask=avfeats_mask)
        #alayerblock2
        hidden_feat,hidden_masks = self.alayerblock2(avfeats,avfeats_mask)
        hidden_length = hidden_masks.squeeze(1).sum(1)

        if isinstance(hidden_feat,tuple):
            hidden_feat = hidden_feat[0]
        if self.normalize_before:
            hidden_feat = self.after_norm(hidden_feat)
            
        return hidden_feat,hidden_length,None

"""
TMCTCEncoder is used for feat vggpreencode 25ps +video 25 ps which is similar to AVConformerEncoder5, but it's lack of a skip connect in audio branch. It is also similar to TM-CTC https://arxiv.org/pdf/1809.02108.pdf
"""
class TMCTCEncoder(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        encoder_conf:dict,
        alayer_num1:int=3,
        vlayer_num1:int=3,
        alayer_num2:int=9,
    ):  
        super().__init__()
        #setting configs
        self._output_size = encoder_conf["hidden_size"]
        conformer_conf = encoder_conf["conformer_conf"]
        embed_conf = encoder_conf["embed_conf"]
        self.conformer_conf = conformer_conf
        self.normalize_before = conformer_conf["normalize_before"]
        pos_enc_layer_type = embed_conf["pos_enc_layer_type"]
        selfattention_layer_type = conformer_conf["selfattention_layer_type"]

        #position embedding layer
        pos_enc_class = AVFineTuneConformerEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)
        self.embed = torch.nn.Sequential(
                pos_enc_class(embed_conf["output_size"], embed_conf["positional_dropout_rate"])
            )
        
        #transformer cross_attention v as q, a as k,v 
        self.fusionblock1 = NewDimConvert(in_channels=encoder_conf["hidden_size"]*2,out_channels=encoder_conf["hidden_size"]) 
        
        # a & v conformer block
        self.alayerblock1 = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=alayer_num1,**conformer_conf) # a 3layer conformer
        self.vlayerblock1 = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=vlayer_num1,**conformer_conf) #a 3-layer conformer
        
        #transformer cross_attention v as q, a as k,v 
        self.fusionblock2 = NewDimConvert(in_channels=encoder_conf["hidden_size"]*2,out_channels=encoder_conf["hidden_size"])  
        self.alayerblock2 = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=alayer_num2,**conformer_conf) #a 9-layer conformer

        if self.normalize_before:
            self.after_norm = LayerNorm(self._output_size)

    def output_size(self) -> int:
        return self._output_size

 
    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). 
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #modal alignment
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        mask = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        #embd
        feats = self.embed(feats)
        if not isinstance(feats,tuple):
            video = self.embed(video) #add abs emb
        else:
            video = (video,feats[1])

        #fusionblock1
        if isinstance(feats,tuple):
            avfeats = self.fusionblock1(torch.cat([feats[0],video[0]],axis=-1))
            avfeats = (avfeats,feats[1])
        else:
            avfeats = self.fusionblock1(torch.cat([feats,video],axis=-1))

        #alayerblock1
        avfeats,avfeats_mask = self.alayerblock1(avfeats,mask)
        #vlayerblock
        video,video_mask = self.vlayerblock1(video,mask)

        #fusionblock2
        if isinstance(avfeats,tuple):
            avfeats_embs = self.fusionblock2(torch.cat([avfeats[0],video[0]],axis=-1))
            avfeats = (avfeats_embs,avfeats[1])
        else:
            avfeats = self.fusionblock2(torch.cat([avfeats,video],axis=-1))

        #alayerblock2
        hidden_feat,hidden_masks = self.alayerblock2(avfeats,avfeats_mask)
        hidden_length = hidden_masks.squeeze(1).sum(1)

        if isinstance(hidden_feat,tuple):
            hidden_feat = hidden_feat[0]
        if self.normalize_before:
            hidden_feat = self.after_norm(hidden_feat)
            
        return hidden_feat,hidden_length,None

"""
TMCTC contain 12 a-conformer , 3 layer conformer there is no fusion between the two model during the encoder block TM-seq2seq encoder https://arxiv.org/pdf/1809.02108.pdf
"""
class TMSeq2SeqEncoder(AVOutEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        encoder_conf:dict,
        alayer_num:int=12,
        vlayer_num:int=3,
    ):  
        super().__init__()
        #setting configs
        self._output_size = encoder_conf["hidden_size"]
        conformer_conf = encoder_conf["conformer_conf"]
        embed_conf = encoder_conf["embed_conf"]
        self.conformer_conf = conformer_conf
        pos_enc_layer_type = embed_conf["pos_enc_layer_type"]
        selfattention_layer_type = conformer_conf["selfattention_layer_type"]
        self.normalize_before = conformer_conf["normalize_before"]

        #position embedding layer
        pos_enc_class = AVFineTuneConformerEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)
        self.embed = torch.nn.Sequential(
                pos_enc_class(embed_conf["output_size"], embed_conf["positional_dropout_rate"])
            )
        
        # a & v conformer block
        self.alayerblock = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=alayer_num,**conformer_conf) # a 3layer conformer
        self.vlayerblock = AVFineTuneConformerEncoder.get_comformerblocks(num_blocks=vlayer_num,**conformer_conf) #a 3-layer conformer
        
        if self.normalize_before:
            self.after_norm = LayerNorm(self._output_size)
            self.after_norm_v = LayerNorm(self._output_size)

    def output_size(self) -> int:
        return self._output_size

 
    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). 
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #modal alignment
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        mask = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        #embd
        feats = self.embed(feats)
        if not isinstance(feats,tuple):
            video = self.embed(video) #add abs emb
        else:
            video = (video,feats[1])

        #alayerblock
        feats,feats_mask = self.alayerblock(feats,mask)
        #vlayerblock
        video,video_mask = self.vlayerblock(video,mask)

        if isinstance(feats,tuple):
            feats = feats[0]
            video = video[0]
        if self.normalize_before:
            feats = self.after_norm(feats)
            video = self.after_norm_v(video)

        feats_lengths = feats_mask.squeeze(1).sum(1)
        video_lengths = video_mask.squeeze(1).sum(1)
        
        return feats,feats_lengths,video,video_lengths, None

"""
AVCrossAttentionEncoder is used for feat vggpreencode 25ps +video 25 ps ,different with other encoder it use audio feat as K, and speech_feat as K,Q, and it will be fine-tune based on a 12-layer asr comferformer
"""
class AVCrossAttentionEncoder(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        v_num_blocks: int = 3,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        crossfusion_num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer="conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        src_first: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        srcattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        attaug: AbsSpecAug = None,
    ):  
        assert check_argument_types()
        super().__init__()
        self.attaug = attaug
        self._output_size = output_size
        if src_first:
            logging.info("using src_first")
        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        pos_enc_class = AVFineTuneConformerEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)

        self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )

        
        vencoder_selfattn_layer,vencoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_selfattn_layer,encoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_srcattn_layer,encoder_srcattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(srcattention_layer_type,
                                                                                pos_enc_layer_type,
                                                                                attention_heads,
                                                                                output_size,
                                                                                zero_triu,
                                                                                attention_dropout_rate)

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        
        #video branch
        if v_num_blocks !=0:
            self.v_encoder = repeat(
                v_num_blocks,
                lambda lnum: EncoderLayer(
                    output_size,
                    vencoder_selfattn_layer(*vencoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        else: 
            self.v_encoder = None
            self.video_norm = LayerNorm(output_size)

        #audio branch
        self.cross_fusion_encoder = torch.nn.ModuleList()
        for i in range(crossfusion_num_blocks):
            self.cross_fusion_encoder.append(
                CrossAttentionFusionEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    encoder_srcattn_layer(*encoder_srcattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    src_first,
                    )
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size
    
    @staticmethod
    def getattentionMHA(
        selfattention_layer_type,
        pos_enc_layer_type,
        attention_heads,
        output_size,
        zero_triu,
        attention_dropout_rate):

        if selfattention_layer_type == "selfattn":
            attn_layer = MultiHeadedAttention
            attn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            attn_layer = RelPositionMultiHeadedAttention
            attn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)
        return attn_layer, attn_layer_args
     
    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        #modal alignment
        feats_lengths = feats_lengths.min(video_lengths)
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        video_masks = masks.clone()

        if self.attaug:
            att_augmasks = self.attaug(video)
            if len(self.cross_fusion_encoder) < len(att_augmasks):
                att_augmasks = att_augmasks[:self.cross_fusion_encoder]
            else:
                att_augmasks.extend([None]*(len(self.cross_fusion_encoder)-len(att_augmasks)))
        else:
            att_augmasks = [None]*len(self.cross_fusion_encoder)
        #posemb
        feats = self.embed(feats) #add posemb
        if not isinstance(feats,tuple): #if absemb
            video = self.embed(video)
        else:
            video = (video,feats[1])
        
        #vencoder
        if self.v_encoder:
            video,video_masks = self.v_encoder(video,video_masks)
        else:
            if isinstance(feats,tuple): #if relemb
                video = video[0]
            video = self.video_norm(video)
    
        #crossfusionencoder
        for cross_fusion_encoder,att_augmask in zip(self.cross_fusion_encoder,att_augmasks):
            feats,masks,video,video_masks = cross_fusion_encoder(feats,masks,video,video_masks,att_augmask)

        if isinstance(feats, tuple):
            feats = feats[0]
        if self.normalize_before:
            feats = self.after_norm(feats)

        olens = masks.squeeze(1).sum(1)
        return feats, olens, None

"""
NewAVCrossAttentionEncoder is used for feat vggpreencode 25ps +video 25 ps ,different with other encoder it use audio feat as K, and speech_feat as K,Q, and it will be fine-tune based on a 12-layer asr comferformer
"""
class NewAVCrossAttentionEncoder(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        v_num_blocks: int = 3,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        crossfusion_num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer="conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        src_first: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        srcattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        attaug: AbsSpecAug = None,
    ):  
        assert check_argument_types()
        assert v_num_blocks!=0
        self.attaug = attaug
        self.v_num_blocks = v_num_blocks
        self.crossfusion_num_blocks = crossfusion_num_blocks
        super().__init__()
        self._output_size = output_size
        if src_first:
            logging.info("using src_first")
        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        pos_enc_class = AVFineTuneConformerEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)

        self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )

        
        vencoder_selfattn_layer,vencoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_selfattn_layer,encoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_srcattn_layer,encoder_srcattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(srcattention_layer_type,
                                                                                pos_enc_layer_type,
                                                                                attention_heads,
                                                                                output_size,
                                                                                zero_triu,
                                                                                attention_dropout_rate)

        self.normalize_before = normalize_before
        positionwise_layer,positionwise_layer_args = AVFineTuneConformerEncoder.get_positionwise_layer(positionwise_layer_type,
                                                                                                        output_size,
                                                                                                        linear_units,
                                                                                                        dropout_rate,
                                                                                                        positionwise_conv_kernel_size)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        
        self.v_encoderlayers = torch.nn.ModuleList()
        for _ in range(v_num_blocks):
            self.v_encoderlayers.append(
                EncoderLayer(
                        output_size,
                        vencoder_selfattn_layer(*vencoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                        convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                )    

        self.vmemory_fusion = NewDimConvert(in_channels=output_size*v_num_blocks,out_channels=output_size)  
        self.vmemory_fusion_norm = LayerNorm(output_size)    

        self.cross_fusion_encoderlayers = torch.nn.ModuleList()
        for _ in range(crossfusion_num_blocks):
            self.cross_fusion_encoderlayers.append(
                CrossAttentionFusionEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                encoder_srcattn_layer(*encoder_srcattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                src_first
            ),)   
    
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size


     
    def forward(self,feats,feats_lengths,video,video_lengths,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        #modal alignment
        feats_lengths = feats_lengths.min(video_lengths)
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        video_masks = masks.clone()

        #posemb
        feats = self.embed(feats) #add posemb
        if not isinstance(feats,tuple): #if absemb
            video = self.embed(video)
        else:
            video = (video,feats[1])
        
        #vencoder + fusion_encoder
        video_memories = []
        for i in range(self.v_num_blocks):
            video,video_masks = self.v_encoderlayers[i](video,video_masks)
            feats,masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video,video_masks)
            if isinstance(feats,tuple): #if absemb
                video_memories.append(video[0])  
            else:
                video_memories.append(video)

        #concat layer
        video_memories = self.vmemory_fusion(torch.cat(video_memories,axis=-1))
        video_memories = self.vmemory_fusion_norm(video_memories)

        #fusion_encoder
        for i in range(self.v_num_blocks,self.crossfusion_num_blocks):
            feats,masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video_memories,video_masks)

        if isinstance(feats, tuple):
            feats = feats[0]
        if self.normalize_before:
            feats = self.after_norm(feats)

        olens = masks.squeeze(1).sum(1)
        return feats, olens, None

class NewAVOutCrossAttentionEncoder(AVOutEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        v_num_blocks: int = 3,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        crossfusion_num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer="conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        srcattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        attaug: AbsSpecAug = None,
    ):  
        assert check_argument_types()
        assert v_num_blocks!=0
        self.attaug = attaug
        self.v_num_blocks = v_num_blocks
        self.crossfusion_num_blocks = crossfusion_num_blocks
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        pos_enc_class = AVFineTuneConformerEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)

        self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )

        
        vencoder_selfattn_layer,vencoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_selfattn_layer,encoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_srcattn_layer,encoder_srcattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(srcattention_layer_type,
                                                                                pos_enc_layer_type,
                                                                                attention_heads,
                                                                                output_size,
                                                                                zero_triu,
                                                                                attention_dropout_rate)

        self.normalize_before = normalize_before
        positionwise_layer,positionwise_layer_args = AVFineTuneConformerEncoder.get_positionwise_layer(positionwise_layer_type,
                                                                                                        output_size,
                                                                                                        linear_units,
                                                                                                        dropout_rate,
                                                                                                        positionwise_conv_kernel_size)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        
        self.v_encoderlayers = torch.nn.ModuleList()
        for _ in range(v_num_blocks):
            self.v_encoderlayers.append(
                EncoderLayer(
                        output_size,
                        vencoder_selfattn_layer(*vencoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                        convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                )    

        self.vmemory_fusion = NewDimConvert(in_channels=output_size*v_num_blocks,out_channels=output_size)  
        self.vmemory_decoder_fusion = NewDimConvert(in_channels=output_size*v_num_blocks,out_channels=output_size)  
        self.vmemory_fusion_norm = LayerNorm(output_size) 
        self.vmemory_decoder_fusion_norm =  LayerNorm(output_size) 

        self.cross_fusion_encoderlayers = torch.nn.ModuleList()
        for _ in range(crossfusion_num_blocks):
            self.cross_fusion_encoderlayers.append(
                CrossAttentionFusionEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                encoder_srcattn_layer(*encoder_srcattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),)   
    
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size


     
    def forward(self,feats,feats_lengths,video,video_lengths,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        #modal alignment
        feats_lengths = feats_lengths.min(video_lengths)
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        video_masks = masks.clone()

        #posemb
        feats = self.embed(feats) #add posemb
        if not isinstance(feats,tuple): #if absemb
            video = self.embed(video)
        else:
            video = (video,feats[1])
        
        #vencoder+fusion_encoder
        video_memories = []
        for i in range(self.v_num_blocks):
            video,video_masks = self.v_encoderlayers[i](video,video_masks)
            feats,masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video,video_masks)
            if isinstance(feats,tuple): #if absemb
                video_memories.append(video[0])  
            else:
                video_memories.append(video)

        #concat video encoder memories for encoder
        video_encoder_memories = self.vmemory_fusion(torch.cat(video_memories,axis=-1))
        video_encoder_memories = self.vmemory_fusion_norm(video_encoder_memories)
        
        #concat video encoder memories for decoder
        video_decoder_memories = self.vmemory_decoder_fusion(torch.cat(video_memories,axis=-1))
        video_decoder_memories = self.vmemory_decoder_fusion_norm(video_decoder_memories)

        #fusion_encoder
        for i in range(self.v_num_blocks,self.crossfusion_num_blocks):
            feats,masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video_encoder_memories,video_masks)

        if isinstance(feats, tuple):
            feats = feats[0]
        if self.normalize_before:
            feats = self.after_norm(feats)

        olens = masks.squeeze(1).sum(1)
        video_lens = video_masks.squeeze(1).sum(1)
        return feats,olens,video_decoder_memories,video_lens, None

class TCNFusionEncoder(AbsEncoder): # [b,T,512]->[b,T,256*3] 
    def __init__(
        self,
        single_input_dim=512,
        fuse_type="tcn",
        hidden_channels=[256 *3, 256 * 3, 256 * 3],
        kernels_size= [3, 5, 7],
        dropout=0.2,
        act_type="prelu",
        downsample_type="norm"
    ):  
        super().__init__()
        fuse_setting = {
            'in_channels': [single_input_dim, single_input_dim],
            "hidden_channels":hidden_channels,
            "kernels_size":kernels_size,
            "dropout":dropout,
            "act_type":act_type,
            "downsample_type":downsample_type,
            }
        self.subsampling = Conv2dSubsampling(
                            single_input_dim,
                            single_input_dim,
                            dropout,
                            None)
        
        self.fusion = AudioVisualFuse(fuse_type=fuse_type, fuse_setting=fuse_setting)
        self.dimturner = DimConvert(in_channels=256*3,out_channels=256)

    def output_size(self) -> int:
        return 256

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #100 fps ->25fps downsampling and alignment
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats_lengths.device)
        feats, masks = self.subsampling(feats, masks)
        feats_lengths = masks.squeeze(1).sum(1)

        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        #fusion TCN
        feats = feats.transpose(1,2)#[B,T,D]->[B,D,T]
        video = video.transpose(1,2)
        hidden_feat, hidden_feat_lengths = self.fusion([feats], [video], feats_lengths) #[B,D,T]->[B,D,T]
        hidden_feat = self.dimturner(hidden_feat.transpose(1,2))
    
        return hidden_feat,hidden_feat_lengths,None

    # print("hhh")
    # fusionnet = TCNFusionEncoder(**dict(  single_input_dim=512,
    #         fuse_type="tcn",
    #         hidden_channels=[256 *3, 256 * 3, 256 * 3],
    #         kernels_size= [3, 5, 7],
    #         dropout=0.2,
    #         act_type="prelu",
    #         downsample_type="norm"))
    # feats = torch.rand(16,90,512)
    # video = torch.rand(16,90,512)
    # feats_l = torch.randint(40,91,(16,))
    # feats_l[0] = 90
    # video_l = torch.randint(90,91,(16,))
    # video_l[0] = 90
    # print("hhh")
    # hidden_feat,hidden_feat_lengths = fusionnet(feats,feats_l,video,video_l)
    # print(hidden_feat.shape,hidden_feat_lengths)

class VConformerEncoder(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        conformer_conf:dict,
        vlayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf) 
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """
        hidden_feat,hidden_feat_lengths,_ = self.vlayer1(video,video_lengths)
        return hidden_feat,hidden_feat_lengths,_


if __name__ == "__main__":
    # yaml_path = "conf/avsrfinetune/avsr_com_finetune.yaml"
    # import yaml 
    # with open(yaml_path) as f :
    #     cfg = yaml.safe_load(f)
    # encoder_conf = cfg["encoder_conf"]
    # encoder = AVFineTuneConformerEncoder(encoder_conf)
    # print(encoder)
    # feats_lengths = torch.randint(128,129,(32,))                                  
    # feats = torch.rand(32,128,512)
    # video_lengths = torch.randint(128,129,(32,))     
    # video = torch.rand(32,128,512)
    # output,olen,_ = encoder(feats,feats_lengths,video,video_lengths)
    # print(output.shape,olen)

    

    # yaml_path = "conf/avsr/AVCrossAttentionEncoder2.yaml"
    # import yaml 
    # with open(yaml_path) as f :
    #     cfg = yaml.safe_load(f)
    # encoder_conf = cfg["encoder_conf"]
    # encoder = AVCrossAttentionEncoder(**encoder_conf)
    # feats_lengths = torch.randint(128,129,(32,))                                  
    # feats = torch.rand(32,128,512)
    # video_lengths = torch.randint(128,129,(32,))     
    # video = torch.rand(32,128,512)
    # output,olen,_ = encoder(feats,feats_lengths,video,video_lengths)
    # print(output.shape,olen)

    # yaml_path = "conf/avsrfinetune/AVTMCTCConformerEncoder0ivsr.yaml"
    # import yaml 
    # with open(yaml_path) as f :
    #     cfg = yaml.safe_load(f)
    # encoder_conf = cfg["encoder_conf"]
    # avlayer_num_conf = cfg["avlayer_num_conf"]
    # encoder = TMCTCEncoder(encoder_conf=encoder_conf,**avlayer_num_conf)
    # feats_lengths = torch.randint(128,129,(32,))                                  
    # feats = torch.rand(32,128,512)
    # video_lengths = torch.randint(128,129,(32,))     
    # video = torch.rand(32,128,512)
    # output,olen,_ = encoder(feats,feats_lengths,video,video_lengths)
    # print(output.shape,olen)


    yaml_path = "conf/avsrfinetune/AVTMSeqConformerEncoder0ivsr.yaml"
    import yaml 
    with open(yaml_path) as f :
        cfg = yaml.safe_load(f)
    encoder_conf = cfg["encoder_conf"]
    avlayer_num_conf = cfg["avlayer_num_conf"]
    encoder = TMSeq2SeqEncoder(encoder_conf=encoder_conf,**avlayer_num_conf)
    max_len = 1
    feats_lengths = torch.randint(max_len,max_len+1,(32,))                                  
    feats = torch.rand(32,max_len,512)
    video_lengths = torch.randint(max_len,max_len+1,(32,))    
    video = torch.rand(32,max_len,512)
    a_output,a_len,v_output,v_len,_ = encoder(feats,feats_lengths,video,video_lengths)
    print(a_output.shape,v_output.shape)


    # yaml_path = "conf/avsrfinetune/newcross/AVffnConformerEncoder0ivsr_lipfmid_triphone02.yaml"
    # import yaml 
    # from espnet2.asr.encoder.avconformer_encoder import NewAVCrossAttentionEncoder
    # import torch
    # with open(yaml_path) as f :
    #     cfg = yaml.safe_load(f)
    # encoder_conf = cfg["encoder_conf"]
    # encoder = NewAVCrossAttentionEncoder(**encoder_conf)
    # max_len = 1
    # feats_lengths = torch.randint(max_len,max_len+1,(32,))                                  
    # feats = torch.rand(32,max_len,512)
    # video_lengths = torch.randint(max_len,max_len+1,(32,))    
    # video = torch.rand(32,max_len,512)
    # feats, olens, _ = encoder(feats,feats_lengths,video,video_lengths)
    # print(feats.shape,olens.shape)
