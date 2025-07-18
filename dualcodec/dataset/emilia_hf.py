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
            self.ds1 = load_dataset(
                "amphion/Emilia-Dataset",
                # data_files={"de": "DE/*.tar"},
                split="train",
                streaming=True,
            )
        else:
            
            en_directory = "/mnt/disks/emilia/emilia_dataset/Emilia/"
            en_paths = [filename for filename in os.listdir(en_directory) if filename.endswith(".tar")]
            language_1 = "EN"
            
            self.ds1 = load_dataset(
                en_directory,
                data_files={language_1.lower(): en_paths},
                split=language_1.lower(),
                num_proc=50,
                cache_dir="/mnt/disks/emilia/emilia_cache",
            )
            
            zh_directory = "/mnt/disks/emilia/emilia_dataset/Emilia/ZH"
            zh_paths = [filename for filename in os.listdir(zh_directory) if filename.endswith(".tar")]
            language_2 = "ZH"
            
            self.ds2 = load_dataset(
                zh_directory,
                data_files={language_2.lower(): zh_paths},
                split=language_2.lower(),
                num_proc=50,
                cache_dir="/mnt/disks/emilia/emilia_cache",
            )
            
            self.dataset = concatenate_datasets([self.ds1, self.ds2])   # type: ignore
            
            
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
