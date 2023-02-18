import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .sparse import DctCompress, SparseEncoder



class AttentionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        assert d_model % heads == 0, "d_model is not divisable by heads"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.dropout = nn.Dropout(dropout)
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        q = self.proj_q(q).view(b, -1, self.heads, self.d_k)
        k = self.proj_q(k).view(b, -1, self.heads, self.d_k)
        v = self.proj_q(v).view(b, -1, self.heads, self.d_k)

        score = torch.einsum("bphd, bqhd -> bhpq", q, k) / math.sqrt(self.d_k)
        if mask is not None:
            score = torch.masked_fill(score, mask != 1, float("-inf"))
        score = self.dropout(F.softmax(score, dim=-1))
        attn = torch.einsum("bhpq, bqhd -> bhpd", score, v)
        attn = rearrange(attn, "b h p d -> b p (h d)")
        return self.linear(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.gelu1 = nn.GELU()
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.gelu1(self.linear1(x))
        x = self.dropout(x)
        return self.gelu2(self.linear2(x))


class ResiduleBlock(nn.Module):
    def __init__(self, fn, d_model, dropout):
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.resweight = nn.Parameter(torch.tensor([0], dtype=torch.float))

    def forward(self, x, *args, **kwargs):
        return x + self.dropout(self.resweight * self.fn(self.layernorm(x), *args, **kwargs))


class Encoder(nn.Module):
    def __init__(self, in_channel, downsample, n_layer, dropout, d_model, heads) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.dropout = dropout

        self.downsample = nn.Conv2d(in_channel, d_model, downsample, downsample)
        self.attns = nn.ModuleList()
        self.ffs = nn.ModuleList()
        for i in range(n_layer):
            self.attns.append(ResiduleBlock(AttentionLayer(d_model, heads, dropout), d_model, dropout))
            self.ffs.append(ResiduleBlock(FeedForward(d_model, dropout), d_model, dropout))

    def forward(self, x):
        x = x.to(torch.float)
        x = F.gelu(self.downsample(x))
        x = rearrange(x, "b c h w -> b (h w) c")
        for attn, ff in zip(self.attns, self.ffs):
            x = attn(x, x, x)
            x = ff(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, n_layer, dropout, d_model, heads, num_cls) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.dropout = dropout
        self.num_cls = num_cls

        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.ffs = nn.ModuleList()
        for i in range(n_layer):
            self.self_attns.append(ResiduleBlock(AttentionLayer(d_model, heads, dropout), d_model, dropout))
            self.cross_attns.append(ResiduleBlock(AttentionLayer(d_model, heads, dropout), d_model, dropout))
            self.ffs.append(ResiduleBlock(FeedForward(d_model, dropout), d_model, dropout))
        self.generator = nn.Linear(d_model, num_cls)

    def forward(self, x, memory):
        device = x.device
        batch_size, seq_len = x.shape[:2]
        tgt_mask = ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), 1)
        tgt_mask = tgt_mask.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1)
        for self_attn, cross_attn, ff in zip(self.self_attns, self.cross_attns, self.ffs):
            x = self_attn(x, x, x, tgt_mask)
            x = cross_attn(x, memory, memory)
            x = ff(x)
        return x, self.generator(x)


class DCTransformer(nn.Module):
    def __init__(self, resolution, block_size, interleave=True, encoder_downsample=2, nlayer=3, d_model=512, nheads=8, chunk_size=896, max_nchunk=10, dropout=0.1) -> None:
        super().__init__()

        self.resolution = resolution
        self.block_size = block_size
        self.interleave = interleave

        h, w = resolution
        chn_ncls = block_size ** 2 * 3 + 3
        pos_ncls = (h // block_size) * (w // block_size) + 3
        val_ncls = 32 * block_size ** 2 + 3

        self.d_model = d_model
        self.encoder_downsample = encoder_downsample
        self.chunk_size = chunk_size
        self.max_nchunk = max_nchunk
        self.decode_size = (block_size ** 2 * 3, h // block_size, w // block_size)
        self.chn_embedding = nn.Embedding(chn_ncls, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_ncls, d_model, padding_idx=0)
        self.val_embedding = nn.Embedding(val_ncls, d_model, padding_idx=0)
        self.chunk_embedding = nn.Embedding(max_nchunk, d_model)
        self.encoder = Encoder(block_size ** 2 * 3, encoder_downsample, nlayer, dropout, d_model, nheads)
        self.chn_decoder = Decoder(nlayer, dropout, d_model, nheads, chn_ncls)
        self.pos_decoder = Decoder(nlayer, dropout, d_model, nheads, pos_ncls)
        self.val_decoder = Decoder(nlayer, dropout, d_model, nheads, val_ncls)

        self.init_parameters()

    def decode(self, chunks, priors, chunk_ids, return_digits=True):
        chn = chunks[:, 0, :] # (batch, seq_len + 1)
        pos = chunks[:, 1, :]
        val = chunks[:, 2, :]

        b, _, h, w = priors.shape 
        memory = self.encoder(priors)

        chn_embed = self.chn_embedding(chn)
        pos_embed = self.pos_embedding(pos)
        val_embed = self.val_embedding(val)

        chunk_id_embedding = self.chunk_embedding(chunk_ids[:, None])

        hidden_chn, logits_chn = self.chn_decoder(chn_embed[:, :-1, :] + pos_embed[:, :-1, :] + val_embed[:, :-1, :] + chunk_id_embedding, memory)

        hidden_pos, logits_pos = self.pos_decoder(hidden_chn + chn_embed[:, 1:, :], memory)

        w_after_downsample = torch.div(w, self.encoder_downsample, rounding_mode="trunc")
        pos_y_in_memory = torch.div(torch.div(pos[:, 1:] - 3, w, rounding_mode="trunc"), self.encoder_downsample, rounding_mode="trunc")
        pos_x_in_memory = torch.div((pos[:, 1:] - 3) % w, self.encoder_downsample, rounding_mode="trunc")

        pos_in_memory = pos_y_in_memory * w_after_downsample + pos_x_in_memory

        pos_mask = pos_in_memory < 0
        pos_in_memory[pos_mask] = 0    # paddings and eos become zeros here, we will take care of this later

        gather_val = torch.gather(memory, 1, pos_in_memory.unsqueeze(-1).repeat(1, 1, self.d_model))
        gather_val = torch.masked_fill(gather_val, pos_mask[:, :, None], value=0)    # paddings and eos are all made zero vectors

        hidden_val, logits_val = self.val_decoder(hidden_pos + gather_val, memory)

        if return_digits:
            return logits_chn, logits_pos, logits_val
        return self.sample_logits(logits_chn).squeeze(-1), self.sample_logits(logits_pos).squeeze(-1), self.sample_logits(logits_val).squeeze(-1)

    def forward(self, chunks, priors, chunk_ids, return_digits=True):
        chunks = F.pad(chunks, (1, 0), mode="constant", value=1)    # add sos

        return self.decode(chunks, priors, chunk_ids, return_digits)

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def sample_logits(self, logits, top_k=None, ignore_idx=(0, 1), do_sample=False):
        if ignore_idx:
            for id in ignore_idx:
                logits[logits == id] = float("-inf")
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next

    @torch.no_grad()
    def greedy(self, chunks, priors, chunk_ids):
        chn = chunks[:, 0, :]
        pos = chunks[:, 1, :]
        val = chunks[:, 2, :]

        b, _, h, w = priors.shape
        memory = self.encoder(priors)

        chn_embed = self.chn_embedding(chn)
        pos_embed = self.pos_embedding(pos)
        val_embed = self.val_embedding(val)

        chunk_id_embedding = self.chunk_embedding(chunk_ids[:, None])

        hidden_chn, logits_chn = self.chn_decoder(chn_embed + pos_embed + val_embed + chunk_id_embedding, memory)

        chn_next = self.sample_logits(logits_chn[:, -1, :])
        chn_next_embed = self.chn_embedding(chn_next)
        chn_embed = torch.cat([chn_embed[:, 1:, :], chn_next_embed], dim=1)

        hidden_pos, logits_pos = self.pos_decoder(hidden_chn + chn_embed, memory)
        pos_next = self.sample_logits(logits_pos[:, -1, :])
        pos = torch.cat([pos[:, 1:], pos_next], dim=1)

        w_after_downsample = torch.div(w, self.encoder_downsample, rounding_mode="trunc")
        
        pos_y_in_memory = torch.div(torch.div(pos - 3, w, rounding_mode="trunc"), self.encoder_downsample, rounding_mode="trunc")
        pos_x_in_memory = torch.div((pos - 3) % w, self.encoder_downsample, rounding_mode="trunc")

        pos_in_memory = pos_y_in_memory * w_after_downsample + pos_x_in_memory

        gather_val = torch.gather(memory, 1, pos_in_memory.unsqueeze(-1).repeat(1, 1, self.d_model))

        hidden_val, logits_val = self.val_decoder(hidden_pos + gather_val, memory)
        val_next = self.sample_logits(logits_val[:, -1, :])

        return chn_next, pos_next, val_next

    @torch.no_grad()
    def chunk_generate(self, priors, chunk_ids):
        device = priors.device
        chunks = torch.ones((priors.size(0), 3, 1), dtype=torch.long, device=device)
        for i in range(self.chunk_size):
            chn_next, pos_next, val_next = self.greedy(chunks, priors, chunk_ids)
            item = torch.cat([chn_next, pos_next, val_next], dim=1)[:, :, None]
            if ((item == 2) | (item == 0)).any():
                return chunks[:, :, 1:], True
            chunks = torch.cat([chunks, item], dim=-1)

        return chunks[:, :, 1:], False
    
    @torch.no_grad()
    def generate(self, prior):
        device = prior.device
        sparser = SparseEncoder(self.block_size, self.interleave)
        for i in range(self.max_nchunk):
            chunk_id = torch.tensor([i], dtype=torch.long, device=device)
            next_chunk, eos = self.chunk_generate(prior, chunk_id)
            next_chunk = sparser.decode(next_chunk, decode_size=self.decode_size)
            mask = prior != 0
            next_chunk = torch.masked_fill(next_chunk, mask, 0)
            prior += next_chunk
            if eos:
                return prior
        return prior


def make_model(
    resolution=(128, 128), 
    block_size=8, 
    q=50, 
    interleave=True, 
    chunk_size=896, 
    max_nchunk=10, 
    encoder_downsample=2, 
    nlayer=3, 
    d_model=512, 
    nheads=8,
    dropout=0.1,
):
    compresser = DctCompress(block_size, q, interleave)
    model = DCTransformer(resolution, block_size, interleave, encoder_downsample, nlayer, d_model, nheads, chunk_size, max_nchunk, dropout)
    return model, compresser
