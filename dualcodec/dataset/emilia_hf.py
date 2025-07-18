# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from datasets import load_dataset, IterableDataset
from tqdm import tqdm
from datasets import concatenate_datasets

path = "DE/*.tar"  # only for testing. please use full data


class EmiliaDataset(IterableDataset):
    def __init__(self, is_debug=True):
        if is_debug:
            self.dataset = load_dataset(
                "amphion/Emilia-Dataset",
                # data_files={"de": "DE/*.tar"},
                split="train",
                streaming=True,
            )
        else:
            # self.dataset = load_dataset("amphion/Emilia-Dataset", streaming=True)['train']

            local_dir = "/mnt/disks/emilia/emilia_dataset/Emilia/ZH"
            tar_paths = [filename for filename in os.listdir(local_dir) if filename.endswith(".tar")]
            max_shards = 10
            language = "EN"
            
            self.dataset = load_dataset(
                local_dir,
                data_files={language.lower(): tar_paths[:max_shards]},
                split=language.lower(),
                num_proc=40,
                cache_dir="/mnt/disks/emilia/emilia_cache",
            )
            
            # local_dir2 = "/mnt/disks/emilia/emilia_dataset/Emilia/ZH"
            # tar_paths_2 = [filename for filename in os.listdir(local_dir) if filename.endswith(".tar")]
            # language = "ZH"
            
            
            
        # self.dataset = self.dataset.map(lambda x: x, remove_columns=["text", "text_id"])
        # self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        # self.dataset = self.dataset.train_test_split(test_size=0.1)
        # self.dataset = self.dataset["train"]

    def __iter__(self):
        for example in self.dataset:
            yield example

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = EmiliaDataset()
    for example in dataset:
        print(example)
        break
    # dataloader with distributed sampler
    import torch
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
