import os

from datasets import load_dataset, IterableDataset
from tqdm import tqdm
import torch
from datasets import concatenate_datasets

path = "DE/*.tar"  # only for testing. please use full data


class EmiliaDataset(IterableDataset):
    def __init__(self, is_debug=True):
        if is_debug:
            # self.dataset = load_dataset(
            #     "amphion/Emilia-Dataset",
            #     # data_files={"de": "DE/*.tar"},
            #     split="train",
            #     streaming=True,
            # )
            
            self.dataset = load_dataset(
                "vanshjjw/amu-pushed-luna-4500r",
                split="train",
            )
            
            samples = []
            for example in self.dataset:
                audio_dict = example["enhanced_audio"]
                waveform = torch.tensor(audio_dict["array"], dtype=torch.float32).unsqueeze(0)
                sr = int(audio_dict["sampling_rate"])
    
                samples.append(
                    {
                        "mp3": {
                            "array": waveform.numpy(),
                            "sampling_rate": sr,
                        },
                    }
                )
                
            self.dataset = samples
            
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
