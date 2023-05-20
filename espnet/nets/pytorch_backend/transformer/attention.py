#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
import torch.nn.functional as F
import numpy
import torch
from torch import nn
from torch.nn import Parameter

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)  #time1,d_k  XXXX  d_k,time1
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class TraditionMultiheadRelativeAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self,num_heads,embed_dim, dropout, kdim=None, vdim=None, bias=True,
                 add_bias_kv=False, add_zero_attn=False, max_relative_position=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        

        #relative position embedding
        self.max_relative_position = max_relative_position
        num_embeddings = self.max_relative_position * 2 + 1
        self.relative_keys_embedding = self.relative_embedding(num_embeddings, self.head_dim)
        self.relative_values_embedding = self.relative_embedding(num_embeddings, self.head_dim)


        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    # relative attention
    def relative_embedding(self, num_embeddings, embedding_dim):
        m = nn.Embedding(num_embeddings, embedding_dim)
        #nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.xavier_uniform_(m.weight, gain=1.)

        return m
    def generate_relative_positions_matrix_bypos(self, rel_pos, max_relative_position):
        with torch.no_grad():
            length = rel_pos.shape[0]
            range_mat = rel_pos.expand(length,length)
            dist_mat= range_mat - range_mat.t()
            dist_mat = torch.clamp(dist_mat, -max_relative_position,
                                   max_relative_position)
            dist_mat = dist_mat + max_relative_position
        return dist_mat        
    def generate_relative_positions_matrix(self, length, max_relative_position):
        with torch.no_grad():
            range_mat = torch.arange(length).expand(length, length)
            dist_mat= range_mat - range_mat.t()
            dist_mat = torch.clamp(dist_mat, -max_relative_position,
                                   max_relative_position)
            dist_mat = dist_mat + max_relative_position
        return dist_mat

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, mask):
        """Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        query = query.transpose(0,1)
        key = key.transpose(0,1)
        value = value.transpose(0,1)
        tgt_len, bsz, embed_dim = query.size() #(T,B,D)
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

 
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)

        q =q*self.scaling

        #shape to mutiple head
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  #(B*H,T,D/H)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) #(B*H,T,D/H)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) #(B*H,T,D/H)

        src_len = k.size(1)

        #QK
        attn_weights = torch.bmm(q, k.transpose(1, 2))  #(B*H,T,D/H) x (B*H,D/H,T) -> (B*H,T,T)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        #generate R
        relative_positions_matrix = self.generate_relative_positions_matrix(src_len, self.max_relative_position).to(attn_weights.device) #[tgt_T,src_T]
        relations_keys = self.relative_keys_embedding(relative_positions_matrix)[-tgt_len:] #(tgt_TQ,src_T,D/H)
        
        #QR
        q_t = q.permute(1,0,2) #(B*H,tgt_T,D/H) -> (tgt_T,B*H,D/H)
        r_t = relations_keys.transpose(1, 2) #(tgt_T,src_T,D/H)-> (tgt_T,D/H,src_T)
        relations_keys_logits = torch.bmm(q_t, r_t)  #(tgt_T,B*H,src_T)
        #QR+QK
        attn_weights += relations_keys_logits.transpose(0, 1) #(B*H,tgt_T,src_T)

        #mask attention
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) #(B,H,T,src_T)
        
        if mask is not None:
            min_value = float(
                    numpy.finfo(torch.tensor(0, dtype=attn_weights.dtype).numpy().dtype).min
                )
            mask = mask.unsqueeze(1).eq(0) #(B,tgt_T,src_T)->#(B,1,tgt_T,src_T)
   
            attn_weights = attn_weights.masked_fill(mask,min_value) #masked_fill must before softmax
            attn_weights = F.softmax(attn_weights, dim=-1).masked_fill(
                    mask, 0.0
                ) #just for security to set they as zero
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)
                
            
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        #A = QK+QR end      
        
        #AV
        attn = torch.bmm(attn_weights, v)  #(B*H,T,src_T) x (B*H,src_T,D/H)->(B*H,tgt_T,D/H)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        #AR
        relations_values = self.relative_values_embedding(relative_positions_matrix)[-tgt_len:] #(tgt_T,src_T,D/H)
        attn_weights_t = attn_weights.permute(1,0,2) #(B*H,T,src_T) (T,B*H,src_T)
        relations_values_attn = torch.bmm(attn_weights_t.float(), relations_values.float()).type_as(attn_weights) #(tgt_T,B*H,src_T) * #(tgt_T,src_T,D/H)
        attn += relations_values_attn.transpose(0, 1) #(tgt_T,B*H,D/H)-> (B*H,tgt_T,D/H)

        #end multiplehead
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn.transpose(0,1)


    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)





# #test tranditionrelateive position mutiple head
# from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
# from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
# ys_in_lens = torch.tensor([10,9,8,7])
# tgt_mask = ~make_pad_mask(ys_in_lens)[:, None, :]
# # m: (1, L, L)
# m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
# # tgt_mask: (B, L, L)
# input_mask = tgt_mask & m
# input_mask = input_mask[:, -1:, :]
# embed_dim = 512
# num_heads = 8
# attetnion = TraditionMultiheadRelativeAttention(embed_dim, num_heads)
# tgt = torch.rand(4,10,512)
# tgt_q = tgt[:, -1:, :]
# print(input_mask.shape)
# output = attetnion(tgt_q,tgt,tgt,input_mask)
# print(output.shape)


#test relative postion embedding matrix and embedding matrix
# def generate_relative_positions_matrix(length, max_relative_position):
#     range_mat = torch.arange(length).expand(length, length)
#     dist_mat= range_mat - range_mat.t()
#     dist_mat = torch.clamp(dist_mat, -max_relative_position,
#                             max_relative_position)
#     dist_mat = dist_mat + max_relative_position
#     return dist_mat

# def relative_embedding(num_embeddings, embedding_dim):
#         m = nn.Embedding(num_embeddings, embedding_dim)
#         return m
# max_relative_position = 64
# k_len = 5
# tgt_len = 0
# num_embeddings = max_relative_position*2-1
# head_dim = 64
# relative_positions_matrix = generate_relative_positions_matrix(k_len,max_relative_position)
# print(relative_positions_matrix)
# print(relative_positions_matrix.shape)
# relative_keys_embedding = relative_embedding(num_embeddings, head_dim) #
# # relations_keys = relative_keys_embedding(relative_positions_matrix)[-tgt_len:]
# relations_keys1 = relative_keys_embedding(torch.tensor([[1,2],[3,4]]))
# print(relations_keys1.shape)
# relations_keys0 = relative_keys_embedding(torch.tensor([1]))
# print(relations_keys0.shape)
# print(any(relations_keys1[0][0]==relations_keys0[0]))
# print(relations_keys.shape)


