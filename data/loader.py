import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset


class RexGradDataset(IterableDataset):
    def __init__(self,split: str,tokenizer, image_transform):
        self.ds=load_dataset("rajpurkarlab/RexGradient-160K",split=split,streaming=True)
        self.tokenizer=tokenizer
        self.image_transform=image_transform
    def __iter__(self):
        for sample in self.ds:
            image=self.image_transform(sample["image"])
            text=sample['report']
            encoding=self.tokenizer(text,truncation=True,padding="max_length",max_lentgh=256)
            yield {
                "input_ids:torch.tensor(encoding["input_ids"]),
                "pixel": image,
                "attention_mask": torch.tensor(enc["attention_mask"])
            }
            
class RexGradDataModule(pl.LightningDataModule):
    def __init__(self, hf_dir: str, batch_size: int=32, num_workers: int=4):
        super().__init__()
        self.hf_dir = hf_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

   def setup(self, stage=None):
        self.train_ds = RexGradientDataset(
            split="train",
            tokenizer=self.tokenizer,
            image_transform=get_image_transform("train"),
        )
       self.val_ds = RexGradientDataset(
            split="validation",
            tokenizer=self.tokenizer,
            image_transform=get_image_transform("eval"),
        )
        self.test_ds = RexGradientDataset(
            split="test",
            tokenizer=self.tokenizer,
            image_transform=get_image_transform("eval"),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
