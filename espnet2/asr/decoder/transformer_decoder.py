# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention,TraditionMultiheadRelativeAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer,AVCrossDecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding,RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder,AVInDecoder
import torch.nn as nn
import math
from espnet2.utils.conv_tbc import LinearizedConv1d

# add a 2-layer conv before send into decoders with trandition relative postion attetion
class ConvTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        conv_out_channels: list=[512,512],#text related
        conv_kernel_sizes: list=[3,3],
        conv_layer_norm: bool=True,
        num_blocks:int=6, #decoder related
        attention_heads: int = 8,
        normalize_before: bool = True,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        concat_after: bool = False,
        self_att_RPE: bool = True, #sefl.att related
        max_relative_position: int = 64,
        tex_emb_dim: int = -1,
        ):  
        assert len(conv_out_channels) == len(conv_kernel_sizes)
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.conv_layer_norm = conv_layer_norm
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        #1.text embeding layer
        if tex_emb_dim > 0:
            self.embed = torch.nn.Embedding(vocab_size, tex_emb_dim)  
        else:
            self.embed = torch.nn.Embedding(vocab_size, attention_dim)     
        #2.text conv layer
        self.conv_layers = torch.nn.ModuleList()

        if conv_out_channels[-1] != attention_dim: 
            conv_out_channels[-1] = attention_dim

        for i in range(len(conv_out_channels)):
            if i == 0:
                if  tex_emb_dim > 0:
                    conv_layer = LinearizedConv1d(tex_emb_dim, conv_out_channels[0], conv_kernel_sizes[0], padding=(conv_kernel_sizes[0] - 1)) 
                else:
                    conv_layer = LinearizedConv1d(attention_dim, conv_out_channels[0], conv_kernel_sizes[0], padding=(conv_kernel_sizes[0] - 1)) 
            else:
                conv_layer = LinearizedConv1d(
                    conv_out_channels[i-1],
                    conv_out_channels[i],
                    conv_kernel_sizes[i],
                    padding=(conv_kernel_sizes[i] - 1),
                )
            self.conv_layers.append(conv_layer)
            if self.conv_layer_norm:
                self.conv_layers.append(LayerNorm(conv_out_channels[i]))
            self.conv_layers.append(nn.ReLU())
  
        #3.encoder hidden outdecoderlayers
        if self_att_RPE:
            self_attn = TraditionMultiheadRelativeAttention(attention_heads, attention_dim, self_attention_dropout_rate , max_relative_position=max_relative_position)
        else: 
            self_attn = MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate)

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                self_attn,
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.output_layer =  torch.nn.Linear(attention_dim, vocab_size, bias=True)
   

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        x = self.embed(tgt).transpose(0,1) #B,T,C -> T,B,C

        for layer in self.conv_layers:
            x = layer(x)
     
        x = x.transpose(0,1) #T,B,C -> B,T,C 
      

        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        
        if self.normalize_before:
            x = self.after_norm(x)
       
        x = self.output_layer(x)

        olens = tgt_mask.sum(1)#have some problem 
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt).transpose(0,1) 
        for layer in self.conv_layers:
            x = layer(x)
        x = x.transpose(0,1) 
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        
        y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

        # used as a decoder scorer during inference stage
    
        # used as a decoder scorer during inference stage
    
    # 1.give a probability distribution of the next char 2. the each layer output of the decoder
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

#add v memory to decoder, pay attention to fix score input     
class ConvAVCrossDecoder(AVInDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        conv_out_channels: list=[512,512],#text related
        conv_kernel_sizes: list=[3,3],
        conv_layer_norm: bool=True,
        num_blocks:int=6, #decoder related
        attention_heads: int = 8,
        normalize_before: bool = True,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        concat_after: bool = False,
        self_att_RPE: bool = True, #sefl.att related
        max_relative_position: int = 64,
        tex_emb_dim: int = -1,
        ):  
        assert len(conv_out_channels) == len(conv_kernel_sizes)
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.conv_layer_norm = conv_layer_norm
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        #1.text embeding layer
        if tex_emb_dim > 0:
            self.embed = torch.nn.Embedding(vocab_size, tex_emb_dim)  
        else:
            self.embed = torch.nn.Embedding(vocab_size, attention_dim)     
        #2.text conv layer
        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(conv_out_channels)):
            if i == 0:
                if  tex_emb_dim > 0:
                    conv_layer = LinearizedConv1d(tex_emb_dim, conv_out_channels[0], conv_kernel_sizes[0], padding=(conv_kernel_sizes[0] - 1)) 
                else:
                    conv_layer = LinearizedConv1d(attention_dim, conv_out_channels[0], conv_kernel_sizes[0], padding=(conv_kernel_sizes[0] - 1)) 
            else:
                conv_layer = LinearizedConv1d(
                    conv_out_channels[i-1],
                    conv_out_channels[i],
                    conv_kernel_sizes[i],
                    padding=(conv_kernel_sizes[i] - 1),
                )
            self.conv_layers.append(conv_layer)
            if self.conv_layer_norm:
                self.conv_layers.append(LayerNorm(conv_out_channels[i]))
            self.conv_layers.append(nn.ReLU())
  
        #3.encoder hidden outdecoderlayers
        if self_att_RPE:
            self_attn = TraditionMultiheadRelativeAttention(attention_heads, attention_dim, self_attention_dropout_rate , max_relative_position=max_relative_position)
        else: 
            self_attn = MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate)

        self.decoders = repeat(
            num_blocks,
            lambda lnum: AVCrossDecoderLayer(
                attention_dim,
                self_attn,
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.output_layer =  torch.nn.Linear(attention_dim, vocab_size, bias=True)
   

    def forward(
        self,
        a_pad: torch.Tensor,
        a_lens: torch.Tensor,
        v_pad: torch.Tensor,
        v_lens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        a_memory = a_pad
        a_memory_mask = (~make_pad_mask(a_lens, maxlen=a_memory.size(1)))[:, None, :].to(
            a_memory.device
        )

        v_memory = v_pad
        v_memory_mask = (~make_pad_mask(v_lens, maxlen=v_memory.size(1)))[:, None, :].to(
            v_memory.device
        )

        x = self.embed(tgt).transpose(0,1) #B,T,C -> T,B,C

        for layer in self.conv_layers:
            x = layer(x)
     
        x = x.transpose(0,1) #T,B,C -> B,T,C 
      

        x, tgt_mask, a_memory, a_memory_mask,v_memory, v_memory_mask = self.decoders(
            x, tgt_mask, a_memory, a_memory_mask,v_memory, v_memory_mask
        )
        
        if self.normalize_before:
            x = self.after_norm(x)
       
        x = self.output_layer(x)

        olens = tgt_mask.sum(1)#have some problem 
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        a_memory: torch.Tensor,
        v_memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            a_memory: encoded a_memory, float32  (batch, maxlen_in, feat)
            v_memory: encoded v_memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt).transpose(0,1) 
        for layer in self.conv_layers:
            x = layer(x)
        x = x.transpose(0,1) 
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, a_memory, a_memory_mask,v_memory, v_memory_mask = decoder(
                x, tgt_mask, a_memory, None, v_memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        
        y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache
         
    # 1.give a probability distribution of the next char 2. the each layer output of the decoder
    def score(self, ys, state, a, v):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=a.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, a.unsqueeze(0),v.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], a_s: torch.Tensor, v_s: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=a_s.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, a_s, v_s, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


class BaseTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        # Must set by the inheritance
        self.decoders = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)#have some problem 
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache
        
    # used as a decoder scorer during inference stage
    # 1.give a probability distribution of the next char 2. the each layer output of the decoder
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class LightweightConvolutionTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        assert check_argument_types()
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class LightweightConvolution2DTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        assert check_argument_types()
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolutionTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        assert check_argument_types()
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolution2DTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        assert check_argument_types()
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

if __name__ == "__main__":
    vocab_size = 3885
    encoder_output_size = 512
    decoder = ConvAVCrossDecoder(vocab_size,encoder_output_size)
    bs = 16
    hidden_len = 50
    src_len = 10
    a_pad = torch.rand(bs,hidden_len,512)
    a_lens = torch.randint(hidden_len,hidden_len+1,[bs,])
    v_pad = torch.rand(bs,hidden_len,512)
    v_lens = torch.randint(hidden_len,hidden_len+1,[bs,])
    ys_in_pad = torch.randint(0,vocab_size,[bs,src_len])
    ys_in_lens = torch.randint(src_len,src_len+1,[bs,])
    output = decoder(v_pad,v_lens,a_pad, a_lens,ys_in_pad,ys_in_lens)
    print(output[0].shape,output[1])
    # vocab_size = 3885
    # encoder_output_size = 512
    # decoder = ConvTransformerDecoder(vocab_size,encoder_output_size)
    # bs = 16
    # hidden_len = 50
    # src_len = 10
    # hs_pad = torch.rand(bs,hidden_len,512)
    # hlens = torch.randint(hidden_len,hidden_len+1,[bs,])
    # ys_in_pad = torch.randint(0,vocab_size,[bs,src_len])
    # ys_in_lens = torch.randint(src_len,src_len+1,[bs,])
    # output = decoder(hs_pad, hlens,ys_in_pad,ys_in_lens)
    # print(output[0].shape,output[1])

# ys_in_lens = torch.randint(4,5,[3,])
# tgt_mask = ~make_pad_mask(ys_in_lens)[:, None, :]
# # print(tgt_mask)
#         # m: (1, L, L)
# m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
# # tgt_mask: (B, L, L)
# tgt_mask = tgt_mask & m
# # print(m)
# print(tgt_mask)
# print(tgt_mask[:, -1:, :])
