import os

from datasets import load_dataset, IterableDataset
from tqdm import tqdm
import torch
from datasets import concatenate_datasets

import numpy as np

def _to_mp3(example):
    audio_dict = example["enhanced_audio"]
    waveform = np.asarray(audio_dict["array"], dtype=np.float32)[None, :]  # add channel dim
    sr = int(audio_dict["sampling_rate"])
    return {
        "mp3": {
            "array": waveform,
            "sampling_rate": sr,
        }
    }


class EmiliaDataset(IterableDataset):
    def __init__(self, is_debug=True):
        if is_debug:
            
            self.dataset = load_dataset(
                "vanshjjw/amu-pushed-luna-4500r",
                split="train",
            )
            
            self.dataset = self.dataset.map(
                _to_mp3,
                remove_columns=self.dataset.column_names,
                num_proc=4,
                desc="Converting enhanced_audio to emilia compatible processor pipeline",
            )
            # Keep only the first 4 000 rows as a Dataset object (not list)
            self.dataset = self.dataset.select(range(4300))
            
        else:

            en_directory = "/mnt/disks/emilia/emilia_dataset/Emilia/EN"
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
