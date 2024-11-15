import os
from pathlib import Path

import torch
import numpy as np
import polars as pl
import pickle

def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{} {}\n".format(query_trackid, " ".join(map(str, result))))

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

class BestCheckpoint():
    def __init__(self, checkpoint_dir: Path, mode='max', force_ckpt_save=False):
        self.dir = checkpoint_dir
        self.mode = mode
        if self.mode == 'max':
            self.metric = -np.inf
        else:
            self.metric = np.inf
        self.force_ckpt_save = force_ckpt_save
        self.model_name = None
        self.optim_name = None

    def _save_and_update(
        self,
        model_checkpoint: torch.nn.Module,
        optim: torch.optim.AdamW,
        ranked_list,
        val_embeds,
        test_embeds,
        sim_map_valid,
        sim_map_test,
        epoch: int,
        metric: float
    ):
        if self.model_name is not None:
            os.remove(self.dir / self.model_name)
            os.remove(self.dir / self.optim_name)
            os.remove(self.dir / self.sub_name)
            os.remove(self.dir / self.val_embeds)
            os.remove(self.dir / self.test_embeds)
            os.remove(self.dir / self.sim_map_valid)
            os.remove(self.dir / self.sim_map_test)
            

        self.model_name = f"model_epoch_{epoch}_metric_{metric:.6f}.pth"
        self.optim_name = self.model_name.replace('model', 'opt') 
        self.sub_name = f"epoch_{epoch}_nDCG_{metric:.4f}.csv"
        self.val_embeds = f"val_embeds_epoch_{epoch}.pickle"
        self.test_embeds = f"test_embeds_epoch_{epoch}.pickle"
        self.sim_map_valid = f"val_sim_map_epoch_{epoch}.pickle"
        self.sim_map_test = f"test_sim_map_epoch_{epoch}.pickle"
        
        torch.save(model_checkpoint.state_dict(), self.dir / self.model_name)
        torch.save(optim.state_dict(), self.dir / self.optim_name)
        
        dump_pickle(val_embeds, self.dir / self.val_embeds)
        dump_pickle(test_embeds, self.dir / self.test_embeds)

        dump_pickle(sim_map_valid, self.dir / self.sim_map_valid)
        dump_pickle(sim_map_test, self.dir / self.sim_map_test)

        save_submission(ranked_list, self.dir / self.sub_name)
        print(f"Saved model to : {self.dir / self.model_name}")
            
    def update(
        self,
        model_checkpoint: torch.nn.Module,
        optim: torch.optim.AdamW,
        ranked_list,
        val_embeds,
        test_embeds,
        sim_map_valid,
        sim_map_test,
        epoch: int,
        metric: float
    ):
        if (
            (self.metric in [-np.inf, +np.inf]) or
            (self.mode == 'max' and metric > self.metric) or
            (self.mode == 'min' and metric < self.metric) or
            (self.force_ckpt_save)
        ):
            self.metric = metric
            self._save_and_update(model_checkpoint, optim, ranked_list, val_embeds, test_embeds, sim_map_valid, sim_map_test, epoch, metric)    