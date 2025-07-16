import os

from datasets import load_dataset, IterableDataset
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
            local_dir = "/mnt/disks/emilia/emilia_dataset/Emilia/EN"
            tar_paths = [filename for filename in os.listdir(local_dir) if filename.endswith(".tar")]
            language = "EN"
            
            self.dataset = load_dataset(
                local_dir,
                data_files={language.lower(): tar_paths[:500]},
                split=language.lower(),
                num_proc=40,
                cache_dir="/mnt/disks/emilia/emilia_cache",
            )

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
