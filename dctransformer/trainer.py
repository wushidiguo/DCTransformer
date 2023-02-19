import math
import time
from functools import partial
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange

from .model import DCTransformer, make_model
from .sparse import DctCompress


@torch.no_grad()
def collate_fn(data, compresser, device, tgt_chunk_size=896, overlap_size=128, max_nchunk=10, random_chunk=True):
    x = []
    for i in range(len(data)):
        img = data[i]
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float)
        x.append(img)
    x = torch.stack(x, dim=0).permute(0, 3, 1, 2).to(device)
    x = compresser(x)
    chunks = []
    priors = []
    chunk_ids = []
    for i in range(len(x)):
        img = x[i]
        seq_len = img.size(-1)
        # assert seq_len > overlap_size, "sequence length is smaller than overlap length"
        total_chunk_num = math.ceil(seq_len / tgt_chunk_size)
        # padding last chunk
        img = F.pad(img, (0, total_chunk_num * tgt_chunk_size - seq_len), mode="constant", value=0)

        chunk_id = np.random.choice(min(total_chunk_num, max_nchunk)) if random_chunk else 0    # sample chunks uniformly
        chunk_ids.append(chunk_id)

        chunks.append(img[:, tgt_chunk_size * chunk_id : tgt_chunk_size * (chunk_id + 1)])
        priors.append(img[:, : tgt_chunk_size * chunk_id + overlap_size])

    chunks = torch.stack(chunks, dim=0)
    priors = compresser.sparse_encoder.decode(priors)
    chunk_ids = torch.tensor(chunk_ids)

    return chunks, priors, chunk_ids.to(device)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, resolution) -> None:
        super().__init__()
        self.is_folder = True
        if isinstance(dataset, torch.utils.data.Dataset):
            self.imgs = dataset
            self.is_folder = False
        else:
            imgfolder = Path(dataset)
            self.imgs = [p for p in imgfolder.iterdir() if p.suffix in [".jpg", ".jpeg", ".png", ".bmp"]]

        self.resolution = resolution
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution),
            torchvision.transforms.CenterCrop(resolution)
        ])

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]) if self.is_folder else self.imgs[i][0]
        return self.transform(img)

    def __len__(self):
        return len(self.imgs)


def rate(step, max_step, start_lr=5e-4, max_lr=0.1, warmup=1000):
    step = min(step, max_step)

    cosine_rate = math.cos(step / (max_step + 0.1) * torch.pi * 0.5) * max_lr / start_lr
    
    return min(cosine_rate, (max_lr / start_lr - 1) * step / warmup + 1)


def run(
    imgfolder, 
    savepath,
    load_saved=False,
    with_cuda = True,
    batch_size=512, 
    resolution=(128, 128), 
    block_size=8, 
    q=50, 
    interleave=True, 
    chunk_size=896, 
    overlap_size=128, 
    max_nchunk=10, 
    encoder_downsample=2, 
    nlayer=3, 
    d_model=512, 
    nheads=8,
    loss_scale=(1, 1, 1),
    dropout=0.1,
    start_lr=5e-4,
    max_lr=0.1,
    warmup=1000,
    clip_grad=1,
    tokens_to_process=1e9,
    log_interval=100
):    
    device = "cuda" if torch.cuda.is_available() and with_cuda else "cpu"
    savepath = Path(savepath)
    save_dir = savepath.parent
    temp_dir = save_dir / "temp" / time.strftime("%H%M%S", time.localtime())
    temp_dir.mkdir(parents=True, exist_ok=True)

    model, compresser = make_model(resolution, block_size, q, interleave, chunk_size, max_nchunk, encoder_downsample, nlayer, d_model, nheads, dropout)
    
    if load_saved and savepath.is_file():
        state_dict = torch.load(savepath)
        model.load_state_dict(state_dict)

    chn_loss_scale, pos_loss_scale, val_loss_scale = (1, 1, 1) if not loss_scale else loss_scale

    _ = model.to(device)
    model = model.train()

    dataset = TrainDataset(imgfolder, resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size, 
        shuffle=True, 
        collate_fn=partial(
            collate_fn,
            compresser=compresser, 
            tgt_chunk_size=chunk_size, 
            overlap_size=overlap_size, 
            max_nchunk=max_nchunk,
            device=device
        ),
    )

    h, w = resolution
    chn_ncls = block_size ** 2 * 3 + 3
    pos_ncls = (h // block_size) * (w // block_size) + 3
    val_ncls = 32 * block_size ** 2 + 3

    tokens_per_batch = batch_size * chunk_size * 3
    max_step = math.ceil(tokens_to_process / tokens_per_batch)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: rate(step, max_step, start_lr, max_lr, warmup))

    model.train()
    epoch = 0
    tokens = 0
    while True:
        total_loss = 0
        for i, data in enumerate(dataloader):
            logits_chn, logits_pos, logits_val = model(*data)
            loss = F.cross_entropy(logits_chn.reshape(-1, chn_ncls), data[0][:, 0, :].reshape(-1), ignore_index=0) * chn_loss_scale
            loss += F.cross_entropy(logits_pos.reshape(-1, pos_ncls), data[0][:, 1, :].reshape(-1), ignore_index=0) * pos_loss_scale
            loss += F.cross_entropy(logits_val.reshape(-1, val_ncls), data[0][:, 2, :].reshape(-1), ignore_index=0) * val_loss_scale
            total_loss += loss.item()
            tokens += tokens_per_batch

            optimizer.zero_grad()

            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            lr_scheduler.step()

            if (i + 1) % log_interval == 0:
                print(f"epoch: {epoch + 1}, batch: {i + 1}, lr: {optimizer.param_groups[0]['lr']:.2e}, loss: {total_loss / (i + 1):.2f}")

        epoch += 1

        state_dict = model.state_dict()
        torch.save(state_dict, temp_dir / f"epoch_{epoch:03d}.pt")

        if tokens >= tokens_to_process:
            break

    state_dict = model.state_dict()
    torch.save(state_dict, savepath)

    print(f"End of Training. Tokens processed: {tokens}. Model saved on epoch{epoch}: {savepath}")