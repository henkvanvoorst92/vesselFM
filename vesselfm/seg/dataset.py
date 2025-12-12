import logging
from typing import Tuple
from pathlib import Path
import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset

from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.data import generate_transforms

logger = logging.getLogger(__name__)


class UnionDataset(Dataset):
    """
    Dataset that accumulates all given datasets.
    """
    def __init__(self, dataset_configs, mode, finetune=False):
        super().__init__()
        # init datasets
        self.finetune = finetune
        self.datasets, probs = [], []
        self.len = 0
        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            paths = sorted(list(data_dir.iterdir())) # ensures that we use same 1-shot sample

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "reader": determine_reader_writer(dataset_config.file_format)(),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs
                }
            )
            probs.append(dataset_config.sample_prop)

        # ensure that probs sum up to 1
        probs = torch.tensor(probs)
        self.probs = probs / probs.sum()
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # sample dataset
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # sample data sample
        while True:
            data_idx = idx if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            sample_id =  dataset["paths"][data_idx]

            img_path = [path for path in sample_id.iterdir() if 'img' in path.name][0]
            mask_path = [path for path in sample_id.iterdir() if 'mask' in path.name][0]

            if dataset['filter_dataset_IDs'] is not None:
                if int(img_path.stem.split("_")[-1]) in dataset['filter_dataset_IDs']:
                    continue

            img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
            mask = dataset['reader'].read_images(str(mask_path))[0].astype(bool)

            transformed = dataset['transforms']({'Image': img, 'Mask': mask})
            return transformed['Image'], transformed['Mask'] > 0


class MyUnionDataset(Dataset):
    """
    Dataset that accumulates all given datasets.
    """

    def __init__(self, dataset_configs, mode, finetune=False):
        super().__init__()
        # init datasets
        self.finetune = finetune
        self.datasets, probs = [], []
        self.len = 0
        for name, dataset_config in dataset_configs.items():
            #data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            imagesTr_dir = Path(dataset_config.path) / "imagesTr"
            labelsTr_dir = Path(dataset_config.path) / "labelsTr"

            #paths = sorted(list(data_dir.iterdir()))  # ensures that we use same 1-shot sample
            paths = sorted([*list(imagesTr_dir.iterdir()), *list(labelsTr_dir.iterdir())])
            if hasattr(dataset_config, 'split_file'):
                with open(dataset_config.split_file, "r") as f:
                    split = json.load(f)[dataset_config.fold]
                if mode in split.keys():
                    selected_ids = split[mode]
                    paths = [p for p in paths if any(sid in p.name for sid in selected_ids)]

            self.len += len(paths)

            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "reader": determine_reader_writer(dataset_config.file_format)(),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs,
                    'random_gt_img_pair': dataset_config.get('random_gt_img_pair', False),
                    'possible_channels': dataset_config.get('possible_channels', None),
                    'mode': mode
                }
            )
            probs.append(dataset_config.sample_prop)

        # ensure that probs sum up to 1
        probs = torch.tensor(probs)
        self.probs = probs / probs.sum()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # sample dataset
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # sample data sample
        while True:
            data_idx = idx if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            sample_id =  os.path.basename(dataset['paths'][dataset_id]).split('_')[0].split('.')[0]

            img_path = [path for path in dataset["paths"] if 'imagesTr' in path.parts and sample_id in path.parts[-1]][0]
            mask_path = [path for path in dataset["paths"] if 'labelsTr' in path.parts and sample_id in path.parts[-1]][0]

            if dataset['filter_dataset_IDs'] is not None:
                if int(img_path.stem.split("_")[-1]) in dataset['filter_dataset_IDs']:
                    continue

            if dataset['possible_channels'] is not None:
                selected_channel = np.random.choice(dataset['possible_channels'])

            img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
            mask = dataset['reader'].read_images(str(mask_path))[0].astype(np.int16)

            #Either random sample channel (train) or return all channels (val/test)
            if dataset['possible_channels'] is not None and img.ndim==4:
                if dataset['mode'] == 'train':
                    chan_ix = np.random.choice(dataset['possible_channels'])
                    img = img[chan_ix, ...]
                    if mask.ndim==4:
                            gt_chan_ix = chan_ix if dataset['random_gt_img_pair'] else  np.random.choice(dataset['possible_channels'])
                            mask = mask[gt_chan_ix,...]
                if dataset['mode'] in ['val', 'test']:
                    images, masks = [], []
                    for chan_ix in dataset['possible_channels']:
                        img_chan = img[chan_ix, ...]
                        if mask.ndim==4:
                            gt_chan_ix = chan_ix
                            mask_chan = mask[gt_chan_ix,...]
                        else:
                            mask_chan = mask
                        transformed = dataset['transforms']({'Image': img_chan, 'Mask': mask_chan})
                        images.append(transformed['Image'])
                        masks.append(transformed['Mask'])
                    return images, masks

            transformed = dataset['transforms']({'Image': img, 'Mask': mask})
            return [transformed['Image']], [transformed['Mask']]



def list2batch_collate_fn(batch):
    """
    """
    images, masks = [], []
    for img,lbl in batch:
        images.append(torch.stack(img, dim=0))
        masks.append(torch.stack(lbl, dim=0))

    return torch.cat(images, dim=0), torch.cat(masks, dim=0)