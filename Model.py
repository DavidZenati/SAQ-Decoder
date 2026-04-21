"""
"SAQ: Stabilizer-Aware Quantum Error Correction Decoder"
Implementation of -
Stage 1: Dual-Stream Representation Construction
Stage 2: Syndrome-Logical Transformer Decoder (SLTD)
Logical-Centric Loss
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from Codes import *

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2

def _bits_to_index(bits: torch.Tensor) -> torch.Tensor:
    weights = 2 ** torch.arange(bits.size(1), device=bits.device)
    return (bits * weights).sum(dim=1).long()

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.SN_norm = LayerNorm(layer.size)
        self.LN_norm = LayerNorm(layer.size)
        if N > 1:
            self.SN_norm2 = LayerNorm(layer.size)
            self.LN_norm2 = LayerNorm(layer.size)

    def forward(self, x2, x3,mask_SN, mask_LN):
        for idx in range(len(self.layers)):
            x2 = self.layers[idx](x2, x2, mask_SN, layer_input='qubit')
            x3 = self.layers[idx](x3, x2, mask_LN, layer_input='logical')

            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x2 = self.SN_norm2(x2)
                x3 = self.LN_norm2(x3)
        return self.SN_norm(x2), self.LN_norm(x3)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm_q = LayerNorm(size)
        self.norm_l = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, layer_input):
        if layer_input == 'qubit':
            return x + self.dropout(sublayer(self.norm_q(x)))
        elif layer_input == 'logical':
            return x + self.dropout(sublayer(self.norm_l(x)))
        else:
            return x + self.dropout(sublayer(self.norm_l(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, x2, mask, layer_input):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x2, x2, mask), layer_input)
        return self.sublayer[1](x, self.feed_forward, layer_input)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

#################################
def diff_GF2_mul(H,x):
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x

    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    return tmp
#################################

#########################
####### Model  ##########
#########################

class SAQ_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(SAQ_Transformer, self).__init__()
        ####
        code = args.code
        self.pc_matrix = code.pc_matrix
        self.logic_matrix = code.logic_matrix
        self.logical_classes = 2**code.k #if args.noise_type=='independent' else 2**(2*code.k)

        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)

        # positional encodings
        self.src_embed_S = torch.nn.Parameter(torch.empty((code.pc_matrix.size(0), args.d_model)))
        self.src_embed_L = torch.nn.Parameter(torch.empty((self.logical_classes, args.d_model)))
        self.global_tok = torch.nn.Parameter(torch.randn(1, 1, args.d_model))

        # ff_hidden_width
        self.lp_head = nn.Sequential(
            nn.Linear(code.m, 4*code.m),
            nn.GELU(),
            nn.Linear(4*code.m, self.logical_classes)
        )

        self.N_size = args.N_dec

        self.decoder = Encoder(EncoderLayer(args.d_model, c(attn), c(ff), dropout), args.N_dec)


        self.oned_embed_LN = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])

        self.oned_embed_SN =  torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])

        self.aux_logical_fc = nn.Sequential(
            nn.Linear(self.logical_classes, code.m),
            nn.GELU(),
            nn.Linear(code.m, code.m),
            nn.GELU(),nn.Linear(code.m, code.n))

        self.out_fc_S = nn.Linear(code.m , code.n)
        self.out_fc_L = nn.Linear(self.logical_classes, self.logical_classes)

        self.get_mask(code)
        if args.no_mask > 0:
            self.src_mask = None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, syndrome):

        # (1) Build two token streams
        out_LP = self.lp_head(syndrome)  # [B,k] logits
        SN = self.src_embed_S.unsqueeze(0) * syndrome.unsqueeze(-1) # [B,m,d]
        LN = self.src_embed_L.unsqueeze(0) * out_LP.unsqueeze(-1)  # [B,k,d]
        g = self.global_tok.expand(SN.size(0), -1, -1)  # [B,1,d]
        SN = torch.cat([g, SN], dim=1)  # [B,m+1,d]

        # (2) Cross-message passing
        emb_SN, emb_LN = self.decoder(SN, LN, self.src_mask_SN, self.src_mask_LN)

        # (3) Output head
        out_S = self.out_fc_S(self.oned_embed_SN(emb_SN[:, 1:, :]).squeeze(-1))
        out_L = self.out_fc_L(self.oned_embed_LN(emb_LN).squeeze(-1))

        return out_S, out_L, out_LP

    def loss(self, out_S, out_L, out_LP, z2):

        target_idx = _bits_to_index(logical_flipped(self.logic_matrix.T, z2))
        loss1 = F.cross_entropy(out_L, target_idx)

        loss2 = F.cross_entropy(out_LP, target_idx)

        bin_fun = torch.sigmoid
        pred_err = z2 * (1 - bin_fun(out_S)) + (1 - z2) * bin_fun(out_S)
        L_pred_err = diff_GF2_mul(self.logic_matrix.T, pred_err)
        loss3 = torch.nn.functional.binary_cross_entropy(L_pred_err.to(z2.device), torch.zeros_like(L_pred_err).to(z2.device))

        return loss1, loss2, loss3


    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask_SN = None
            self.src_mask_LN = None
            return

        def build_mask_SN(code):
            m = code.pc_matrix.size(0)
            # local neighbourhood matrix (m × m)
            loc = (code.pc_matrix.float() @ code.pc_matrix.float().T) > 0
            loc.fill_diagonal_(1.)

            # add one row/col for the global token
            star = torch.zeros(m + 1, m + 1, dtype=torch.bool)
            star[1:, 1:] = loc  # keep your 1-hop edges
            star[0, :] = 1  # g attends everyone
            star[:, 0] = 1  # everyone attends g
            star[0, 0] = 1

            return ~star.unsqueeze(0).unsqueeze(0)  # [1,1,m+1,m+1] with True = "mask out"

        def build_mask_LN(code):
            mask_LN = torch.ones(self.logical_classes, code.m + 1)
            src_mask = ~ (mask_LN > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask_SN = build_mask_SN(code)
        src_mask_LN = build_mask_LN(code)
        self.register_buffer('src_mask_SN', src_mask_SN)
        self.register_buffer('src_mask_LN', src_mask_LN)

############################################################
############################################################

if __name__ == '__main__':
    pass