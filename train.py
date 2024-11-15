import warnings
warnings.filterwarnings('ignore')

import ast
import os
import os.path as osp
import signal
import time
import gc
import random
import argparse
import tqdm
import importlib
import socket
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, RandomSampler

import numpy as np
import pandas as pd
import polars as pl
from timm.scheduler import CosineLRScheduler

from configs.config_0 import CFG
from local_dataset.simple_dataset import SimpleDataset
from models.finformer import YCModel
from utils import batch_to_device
from utils.handlers import BestCheckpoint
from utils.torchmetrics import CE

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloaders(cfg: CFG, dataset_class, rank):
    train_ds: Union[SimpleDataset, ConcatDataset]
    val_ds: SimpleDataset
    test_ds: SimpleDataset

    if cfg.do_train:
        train_ds = dataset_class(cfg, mode='train')
    val_ds = dataset_class(cfg, mode='val')
    test_ds = dataset_class(cfg, mode='test')
    
    train_sampler = RandomSampler(train_ds) if cfg.do_train else None
    val_sampler = SequentialSampler(val_ds)
    test_sampler = SequentialSampler(test_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        num_workers=cfg.num_workers,
        sampler=train_sampler,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    ) if cfg.do_train else None

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.val_bs,
        num_workers=cfg.num_workers,
        sampler=val_sampler,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.val_bs,
        num_workers=cfg.num_workers,
        sampler=test_sampler,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dl, val_dl, test_dl, test_ds

def ce(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def vec_translate(a, d):    
    return np.vectorize(d.__getitem__)(a)

@torch.inference_mode()
def get_embeddings(model: YCModel, dataloader, device: str = 'cuda:0'):
    embeds = dict()
    for data in tqdm.tqdm(dataloader):
        data = batch_to_device(data)
        with autocast(enabled=True):
            embeddings = model.extract(data['sample'], data['mask'])
            embeddings = F.normalize(embeddings, dim=1).detach().cpu().numpy()
        for trackid, embedding in zip(data['track_id'].cpu().numpy(), embeddings):
            embeds[trackid] = embedding
    return embeds

@torch.inference_mode()
def get_ranked_list_exact(embeds, top_size: int = 100, device: str = 'cuda:0', similarity_map=False):
    num_chunks = 512
    index2track = list(embeds.keys())
    embeds = np.stack(list(embeds.values()), axis=0)
    embeds = torch.from_numpy(embeds).to(device)
    ranked_list = dict()
    track_id_index = 0
    similarity_all = []
    for chunk in tqdm.tqdm(embeds.chunk(num_chunks)):
        chunk = chunk.to(device)
        cos_sim_chunk_values = torch.mm(chunk, embeds.transpose(0, 1))
        
        cos_sim_chunk_values, cos_sim_chunk = cos_sim_chunk_values.sort(dim=1, descending=True)
        cos_sim_chunk = cos_sim_chunk[:, 1:top_size + 50].detach().cpu().numpy()
        cos_sim_chunk_values = cos_sim_chunk_values[:, 1:top_size+ 50]
        for similarity_index, similarity in zip(cos_sim_chunk, cos_sim_chunk_values):
            gallery_mask = similarity_index != track_id_index
            similarity_index = similarity_index[gallery_mask]
            similarity = similarity[gallery_mask]
            current_trackid = index2track[track_id_index] 
            ranked_list[current_trackid] = vec_translate(similarity_index[:top_size], index2track)
            similarity_all.append(similarity[:top_size].detach().cpu().numpy())
            track_id_index += 1
    if similarity_map:
        similarity_all = np.stack(similarity_all)
        return ranked_list, similarity_all
    return ranked_list

def position_discounter(position):
    return 1.0 / np.sqrt(position)

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg


def eval_submission(submission, gt_meta_info, top_size = 100):
    # track2artist_map = gt_meta_info.set_index('track_id')['clique'].to_dict()
    track2artist_map = {}

    for _, row in gt_meta_info.iterrows():
        clique = row['clique']
        versions = row['versions']    
        for version in versions:
            track2artist_map[version] = clique
            
    artist2tracks_map = gt_meta_info.set_index('clique')['versions'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm.tqdm(submission.keys()):
        query_trackid = int(query_trackid)
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg/ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)


def main_worker(cfg: CFG, config_name):
    print(f"Running normal training")
    
    seed_everything(cfg.seed)
    
    rank = int(cfg.device.split(':')[-1])
    current_device = cfg.device
        
    torch.cuda.set_device(rank)
    dataset_class = importlib.import_module(f"local_dataset.{cfg.dataset}".replace('.py', '')).SimpleDataset
    model_class = importlib.import_module(f"models.{cfg.model}".replace('.py', '')).YCModel
    
    
    df = pd.read_csv('data.csv')
    tmp = np.sqrt(1 / np.sqrt(df[df['label'] != -1]['label'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * cfg.margin_slope + cfg.margin_offset
    if cfg.label_flip_prob > 1e-6:
        margins = np.stack([margins, margins]).flatten()
    cfg.margins = margins
    model: YCModel = model_class(cfg).to(rank)
    
    my_list = ['arcface_head.weight']
    params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))

    head_params = [p[1] for p in params]
    rest_params = [p[1] for p in base_params]

    if cfg.optim == 'AdamW':
        # opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        opt = torch.optim.AdamW(
            [
                {"params" : head_params, "lr" : cfg.lr * cfg.head_lr_multiple},
                {"params" : rest_params, "lr" : cfg.lr}

            ], lr=cfg.lr, weight_decay=cfg.wd
        )
    elif cfg.optim == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    else:
        raise NotImplementedError(f"optimizer {cfg.optim} is not supported")

    if cfg.weights != '':
        sd = torch.load(cfg.weights, map_location=torch.device('cpu'))
        sd = {k.replace('module.', ''):v for k,v in sd.items()}
        print(model.load_state_dict(sd, strict=False))
        opt_sd = torch.load(cfg.weights.replace('model_', 'opt_'), map_location=torch.device('cpu'))
        if cfg.load_opt:
            print(opt.load_state_dict(opt_sd))
        for g in opt.param_groups:
            g['lr'] = cfg.lr
            g['initial_lr'] = cfg.lr
        
    scaler = GradScaler()

    train_dl, val_dl, test_dl, test_ds = create_dataloaders(cfg, dataset_class, rank)
    num_steps_per_epoch = len(train_dl) if cfg.do_train else 0
    if cfg.use_scheduler and cfg.do_train:
        warmup = cfg.epochs_warmup * num_steps_per_epoch
        nsteps = cfg.epochs * num_steps_per_epoch 
        sched = CosineLRScheduler(
            opt, 
            warmup_t=warmup if not cfg.use_t0 else 0, 
            warmup_lr_init=0.0, 
            warmup_prefix=not cfg.use_t0,
            t_initial=(nsteps - warmup) if not cfg.use_t0 else cfg.T0,
            lr_min=cfg.lr_min,
            cycle_limit=1 if not cfg.use_t0 else 123456
        )
    else:
        sched = None

    if cfg.criterion == 'ce':
        criterion = ce
        loss_class = CE
    else:
        raise NotImplementedError(f"criterion : {cfg.criterion}")

    exps_dir = Path(cfg.exp_dir)
    if cfg.debug:
        exps_dir = Path(str(exps_dir) + '_DEBUG')
    
    exp_dir = exps_dir / f"exp_{config_name.split('_')[-1].split('.')[0]}"
    weight_dir = exp_dir / 'weights'
    artifacts_dir = exp_dir / 'artifacts'
    
    if rank == 0:
        summary_writer = SummaryWriter(log_dir=exp_dir)
        if not osp.isdir(cfg.exp_dir):
            os.makedirs(cfg.exp_dir)
        if not osp.isdir(artifacts_dir):
            os.makedirs(artifacts_dir)
        ckpt_handler = BestCheckpoint(weight_dir, mode=cfg.checkpoint_mode)
        # print(model)

    accum_steps = 0
    for epoch in range(1, cfg.epochs + 1):
        if cfg.do_train:
            train_pbar = tqdm.tqdm(train_dl) if rank == 0 else train_dl
            train_loss = loss_class()
            
            for t_iter, batch in enumerate(train_pbar):
                if t_iter >= cfg.max_steps_per_epoch:
                    break
                global_iter = (epoch - 1) * num_steps_per_epoch + t_iter
                batch = batch_to_device(batch, device=current_device)
                
                with autocast(enabled=cfg.enable_amp, dtype=cfg.autocast_dtype):
                    preds = model(batch['sample'], batch['label'], batch['mask'])
                    loss = criterion(preds, batch['label'])

                if global_iter < 100 or loss.item() <= cfg.loss_threshold:
                    scaler.scale(loss).backward()#
                    accum_steps += 1
                    if accum_steps % cfg.grad_accum_steps == 0:
                        if cfg.max_grad_norm is not None:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        scaler.step(opt)
                        scaler.update()
                        if cfg.use_scheduler:
                            sched.step(global_iter)
                        accum_steps = 0
                        opt.zero_grad(set_to_none=cfg.set_grad_to_none)
                else:
                    scaler.scale(loss).backward()
                    opt.zero_grad(set_to_none=cfg.set_grad_to_none)
                
                train_loss.accumulate(preds, batch['label'])
                
                if rank == 0:
                    lr = opt.param_groups[0]['lr']
                    train_pbar.set_description(f"loss : {loss.item():.4f}, lr : {lr:.5f}")
                    summary_writer.add_scalar("lr", lr, global_iter)
            
            if rank == 0:
                summary_writer.add_scalar('loss/train', train_loss.value().item(), epoch)
                
        ## use single gpu for eval, since can't use gather for strings (required by comp. sub format)
        df = pd.read_csv('data.csv')
        df_valid = df[df['split'] == 'val'].reset_index(drop=True)
        valid_cliques = pd.read_csv(cfg.storage_dir / 'cliques2versions.tsv', sep='\t')
        valid_cliques = valid_cliques[valid_cliques['clique'].isin(df_valid['clique'])].reset_index(drop=True)
        valid_cliques['versions'] = valid_cliques['versions'].apply(ast.literal_eval)
        if (epoch - 1) % cfg.do_eval_every == 0 and rank == 0:
            model.eval()
            ###
            val_embeds = get_embeddings(model, val_dl)
            ranked_list, sim_map_valid = get_ranked_list_exact(val_embeds, top_size=100, similarity_map=True)
            print("Calculating the metric")
            nDCG = eval_submission(ranked_list, valid_cliques)
            
            summary_writer.add_scalar('nDCG/val', nDCG, epoch)
            if not osp.isdir(weight_dir):
                os.makedirs(weight_dir)
            ###

            ###
            test_embeds = get_embeddings(model, test_dl)
            ranked_list, sim_map_test = get_ranked_list_exact(test_embeds, top_size=100, similarity_map=True)
            # ###
            if cfg.save_checkpoints:
                print(f"Creating test .parquet file...")
                ckpt_handler.update(
                    model,
                    opt,
                    ranked_list,
                    val_embeds,
                    test_embeds,
                    sim_map_valid,
                    sim_map_test,
                    epoch,
                    nDCG
                )
                print(f"Done!")
            model.train()

        if cfg.debug and epoch >= 5:
            break
        if not cfg.do_train:
            exit()
    
    if rank == 0:
        ds_name = f"yc2024-music-exp{config_name.split('_')[1].split('.')[0]}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config')
    args = parser.parse_args()
    
    config_name = args.config
    cfg: CFG = importlib.import_module(f'configs.{config_name}'.replace('.py', '')).cfg

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'cuda:{i} - {torch.cuda.get_device_name(i)}')
    else:
        print('CUDA is not available')

    main_worker(cfg, config_name)
