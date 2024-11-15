import pandas as pd
import numpy as np
import torch

from configs.config_0 import CFG

class SimpleDataset:
    def __init__(self, cfg: CFG, *, mode='train'):
        self.cfg = cfg
        self.mode = mode

        if mode not in ['train', 'val', 'all', 'test']:
            raise ValueError(f"mode : {mode} is not supported")
        df = pd.read_csv('data.csv')
        if mode == 'train' and self.cfg.use_all_data:
            df = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
        else:
            df = df[df['split'] == mode].reset_index(drop=True)
        print(f"mode : {mode}, train samples : {len(df)}")
        self.df = df
        self.labels = torch.tensor(self.df['label']).long()
        self.transforms = cfg.train_transforms if mode == 'train' else None

    def __getitem__(self, index):
        d = self.df.iloc[index]
        arr = np.load(d['file'])
        track_id = d['track_id']
        clique = d['clique']
        label = self.labels[index]
        arr: np.ndarray
        ## True ~ mask token, hence set all False
        if self.mode in ['train', 'all']:
            # k = np.random.choice([0, 1, 2, 3], size=1)[0]
            # arr = np.rot90(arr, k=k)
            if np.random.rand() < self.cfg.flip_prob:
                arr = arr[::-1, :]
            if np.random.rand() < self.cfg.flip_prob:
                arr = arr[:, ::-1]
            arr = arr.copy()
            if self.transforms is not None:
                h, w = arr.shape

                arr = arr.reshape(h, w, 1)
                arr = self.transforms(image=arr)['image'].astype(np.float32)
                arr = arr.reshape(h, w)
            if np.random.rand() < self.cfg.label_flip_prob:
                ## we can flip the audio to obtain "different" clique
                ## we also change the label to the (index in the orig dataframe + offset by num_classes)
                ## we divide by two, since we expand model prediction space by 2x
                arr = arr[:, ::-1].copy()
                label += (self.cfg.num_classes // 2)
            
        mask = np.zeros((arr.shape[1], )).astype(bool)
        return {
            'sample' : arr,
            'track_id' : track_id,
            'clique' : clique,
            'label' : label,
            'mask' : mask,
        }

    def __len__(self):
        return len(self.df)