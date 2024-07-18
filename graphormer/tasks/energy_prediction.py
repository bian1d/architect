# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Sequence, Union

import pickle
from functools import lru_cache

import lmdb

import numpy as np
import torch
from torch import Tensor
from fairseq.data import (
    FairseqDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

from ..data.dataset import EpochShuffleDataset, GraphormerDataset

import sys
import os
import importlib
from ..data import DATASET_REGISTRY

class LMDBDataset:
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.user_data_dir != "":
            self.__import_user_defined_datasets(cfg.user_data_dir)
            if cfg.dataset_name in DATASET_REGISTRY:
                dataset_dict = DATASET_REGISTRY[cfg.dataset_name]
                self.dm = GraphormerDataset(
                    dataset=dataset_dict["dataset"],
                    dataset_source=dataset_dict["source"],
                    train_idx=dataset_dict["train_idx"],
                    valid_idx=dataset_dict["valid_idx"],
                    test_idx=dataset_dict["test_idx"],
                    seed=cfg.seed)
            else:
                raise ValueError(f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}")
        else:
            self.dm = GraphormerDataset(
                dataset_spec=cfg.dataset_name,
                dataset_source=cfg.dataset_source,
                seed=cfg.seed,
            )
        # import pdb; pdb.set_trace()

    def __import_user_defined_datasets(self, dataset_dir):
        dataset_dir = dataset_dir.strip("/")
        module_parent, module_name = os.path.split(dataset_dir)
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        for file in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, file)
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
            ):
                task_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(module_name + "." + task_name)

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            relaxed_energy=data["y_relaxed"],  # python float
        )


class PBCDataset:
    def __init__(self, dataset: LMDBDataset):
        self.dataset = dataset
        self.cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data["pos"]
        pos_relaxed = data["pos_relaxed"]
        cell = data["cell"]
        atoms = data["atoms"]
        tags = data["tags"]

        offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        expand_pos_relaxed = (
            pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets
        ).view(-1, 3)
        src_pos = pos[tags > 1] if self.filter_by_tag else pos

        dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
        used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
            self.n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        used_expand_tags = tags.repeat(self.n_cells)[
            used_mask
        ]  # original implementation use zeros, need to test
        return dict(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]]),
            tags=torch.cat([tags, used_expand_tags]),
            real_mask=torch.cat(
                [
                    torch.ones_like(tags, dtype=torch.bool),
                    torch.zeros_like(used_expand_tags, dtype=torch.bool),
                ]
            ),
            deltapos=torch.cat(
                [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
            ),
            relaxed_energy=data["relaxed_energy"],
        )


def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


class AtomDataset(FairseqDataset):
    def __init__(self, dataset, keyword):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.atom_list = list(range(-20, 20))
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx)
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        atoms: Tensor = self.dataset[index][self.keyword]
        return self.atom_mapper[atoms]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return pad_1d(samples)


class KeywordDataset(FairseqDataset):
    def __init__(self, dataset, keyword, is_scalar=False, pad_fill=0):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.is_scalar = is_scalar
        self.pad_fill = pad_fill

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index][self.keyword]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.is_scalar:
            return torch.tensor(samples)
        return pad_1d(samples, fill=self.pad_fill)


@register_task("energy_prediction")
class EnergyPredictionTask(FairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", metavar="FILE", help="directory for data")

    @property
    def target_dictionary(self):
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        assert split in [
            "train",
            "valid",
            "test",
        ], "invalid split: {}!".format(split)
        print(" > Loading {} ...".format(split))

        db_path = str(Path(self.cfg.data) / split / "data.lmdb")
        lmdb_dataset = LMDBDataset(db_path)
        pbc_dataset = PBCDataset(lmdb_dataset)

        atoms = AtomDataset(pbc_dataset, "atoms")
        tags = KeywordDataset(pbc_dataset, "tags")
        real_mask = KeywordDataset(pbc_dataset, "real_mask")

        pos = KeywordDataset(pbc_dataset, "pos")

        relaxed_energy = KeywordDataset(pbc_dataset, "relaxed_energy", is_scalar=True)
        deltapos = KeywordDataset(pbc_dataset, "deltapos")

        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pos": pos,
                    "atoms": atoms,
                    "tags": tags,
                    "real_mask": real_mask,
                },
                "targets": {
                    "relaxed_energy": relaxed_energy,
                    "deltapos": deltapos,
                },
            },
            sizes=[np.zeros(len(atoms))],
        )

        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(atoms),
                seed=self.cfg.seed,
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))
        self.datasets[split] = dataset