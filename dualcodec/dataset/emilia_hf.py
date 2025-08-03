import os

from datasets import load_dataset, concatenate_datasets, IterableDataset
from tqdm import tqdm


class EmiliaDataset(IterableDataset):
    def __init__(self, is_debug=True):
        if is_debug:
            self.dataset = load_dataset(
                "amphion/Emilia-Dataset",
                split="train",
                streaming=True,
            )
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
        # (Pdb) example['json']
        # {'dnsmos': 3.4671, 'duration': 17.623, 'id': 'DE_B00000_S00000_W000000', 'language': 'de', 'speaker': 'DE_B00000_S00000', 'text': ' Herzlich Willkommen zu den DSA Nachrichten in 21 Minuten, präsentiert von Hintern im Auge Deiner wöchentlichen Nachrichtensendung rund um DSA, die schwarze Katze und Aventuria. Die heutigen Themen sind ein Sonnenküste Update, die PDFs der zehnten Welle des Collectors Club sind erschienen', 'wav': 'DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000000.mp3'}
        # (Pdb) example['mp3']
        # {'path': 'DE_B00000_S00000_W000000.mp3', 'array': array([-1.80987281e-05, -2.03079671e-05, -2.45754873e-05, ...,
        #         1.24948844e-03,  8.32957274e-04,  2.67164229e-04]), 'sampling_rate': 24000}

        # example['mp3']['array'].shape: [T,]
        break
    # dataloader with distributed sampler
    import torch
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
