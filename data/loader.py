import warnings
from typing import Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .preprocessing import get_image_transforms


class HuggingFaceRadiologyDataset(Dataset):
    
    def __init__(
        self, 
        hf_name: str, 
        split: str = 'train',
        max_samples: Optional[int] = None,
        tokenizer = None,
        image_transform = None,
        text_field: str = 'report',
        image_field: str = 'image',
        streaming: bool = False
    ):
        self.hf_name = hf_name
        self.streaming = streaming
        
        self.ds = load_dataset(hf_name, split=split, streaming=streaming)
        
        if not streaming and max_samples:
            self.ds = self.ds.select(range(min(len(self.ds), max_samples)))
        
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.text_field = text_field
        self.image_field = image_field
        
        self._build_index()
    
    def _build_index(self):
        self.rows = []
        
        if self.streaming:
            print("Using streaming mode")
            return
        
        print(f"Indexing {len(self.ds)} samples...")
        
        for idx, ex in enumerate(self.ds):
            img_field = (
                ex.get(self.image_field) or 
                ex.get('image_path') or 
                ex.get('images')
            )
            
            if img_field and ex.get(self.text_field):
                self.rows.append(ex)
            
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} samples...")
        
        if len(self.rows) == 0:
            raise ValueError(
                f"No valid samples found with both image and '{self.text_field}'"
            )
        
        print(f"Found {len(self.rows)} valid samples")

    def __len__(self):
        if self.streaming:
            return 1000000
        return len(self.rows)

    def _load_image(self, image_spec) -> Image.Image:
        try:
            if isinstance(image_spec, dict):
                if 'path' in image_spec:
                    return Image.open(image_spec['path']).convert('RGB')
                elif 'bytes' in image_spec:
                    import io
                    return Image.open(io.BytesIO(image_spec['bytes'])).convert('RGB')
            elif isinstance(image_spec, str):
                return Image.open(image_spec).convert('RGB')
            elif isinstance(image_spec, Image.Image):
                return image_spec.convert('RGB')
            else:
                return Image.open(image_spec).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'Failed to load image: {e}')

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.streaming:
            ex = next(iter(self.ds.skip(idx).take(1)))
        else:
            ex = self.rows[idx]
        
        img_field = (
            ex.get(self.image_field) or 
            ex.get('image_path') or 
            ex.get('images')
        )
        
        if isinstance(img_field, list):
            img_field = img_field[0]
        
        try:
            image = self._load_image(img_field)
        except Exception as e:
            warnings.warn(f"Failed to load image at index {idx}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        text = ex.get(self.text_field, "")
        if not text:
            text = "[EMPTY]"
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=256, 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return image, input_ids, attention_mask


class TextOnlyDataset(Dataset):
    
    def __init__(
        self, 
        hf_name: str,
        split: str = 'train', 
        tokenizer = None,
        text_field: str = 'report', 
        label_field: str = 'icd_label',
        max_samples: Optional[int] = None,
        max_length: int = 512
    ):
        self.ds = load_dataset(hf_name, split=split)
        if max_samples:
            self.ds = self.ds.select(range(min(len(self.ds), max_samples)))
        
        self.rows = [
            ex for ex in self.ds 
            if ex.get(text_field) and ex.get(label_field) is not None
        ]
        
        if len(self.rows) == 0:
            raise ValueError(
                f"No valid samples with '{text_field}' and '{label_field}'"
            )
        
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.max_length = max_length
        
        print(f"Loaded {len(self.rows)} labeled samples for {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.rows[idx]
        
        text = ex[self.text_field]
        if not text:
            text = "[EMPTY]"
        
        enc = self.tokenizer(
            text,
            truncation=True, 
            padding='max_length',
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        label = torch.tensor(ex[self.label_field], dtype=torch.long)
        
        return input_ids, attention_mask, label


class RadiologyDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        hf_name: str, 
        tokenizer,
        batch_size: int = 32,
        num_workers: int = 4, 
        max_samples: Optional[int] = None,
        stage: str = 'pretrain',
        val_split: float = 0.1,
        test_split: float = 0.1,
        streaming: bool = False
    ):
        super().__init__()
        self.hf_name = hf_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.stage = stage
        self.val_split = val_split
        self.test_split = test_split
        self.streaming = streaming

    def setup(self, stage: Optional[str] = None):
        transform = get_image_transforms(self.stage)
        
        text_field = self._infer_text_field()
        
        full_ds = HuggingFaceRadiologyDataset(
            self.hf_name, 
            split='train',
            max_samples=self.max_samples,
            tokenizer=self.tokenizer, 
            image_transform=transform,
            text_field=text_field,
            streaming=self.streaming
        )
        
        if not self.streaming:
            total_size = len(full_ds)
            test_size = int(total_size * self.test_split)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size - test_size
            
            self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
                full_ds, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")
        else:
            self.train_ds = full_ds
            self.val_ds = full_ds
            self.test_ds = full_ds
    
    def _infer_text_field(self) -> str:
        name_lower = self.hf_name.lower()
        if 'rexgradient' in name_lower or 'mimic' in name_lower:
            return 'report'
        elif 'roco' in name_lower:
            return 'caption'
        else:
            return 'report'

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
