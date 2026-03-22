"""
dataset.py — Chargement, scan et split du dataset de pièces de monnaie
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.bmp', '.webp',
    '.tiff', '.tif', '.pgm', '.ppm', '.pnm',
    '.JPG', '.JPEG', '.PNG', '.TIFF'
]


def find_image(img_dir: str, img_id) -> str | None:
    """Cherche une image avec toutes les extensions possibles."""
    for ext in EXTENSIONS:
        p = os.path.join(img_dir, str(img_id) + ext)
        if os.path.exists(p):
            return p
    return None


def scan_valid_images(csv_path: str, img_dir: str) -> pd.DataFrame:
    """
    Parcourt tout le CSV, vérifie chaque image et retourne
    uniquement les lignes dont l'image est valide et lisible.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values('Id').reset_index(drop=True)  # ordre stable

    valid_indices = []
    n_skipped = 0

    print(f"Scan de {len(df)} images...")
    for idx, row in df.iterrows():
        path = find_image(img_dir, row['Id'])
        if path is None:
            n_skipped += 1
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            valid_indices.append(idx)
        except Exception:
            n_skipped += 1

    print(f"✓ Images valides : {len(valid_indices)} | ✗ Ignorées : {n_skipped}")
    return df.iloc[valid_indices].reset_index(drop=True)


class CoinDataset(Dataset):
    """Dataset PyTorch pour la classification de pièces de monnaie."""

    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform=None, classes: list = None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = classes if classes is not None \
            else sorted(df['Class'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.class_to_idx[row['Class']]
        path = find_image(self.img_dir, row['Id'])

        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            # Image corrompue au runtime → image noire de remplacement
            dummy = Image.new('RGB', (227, 227), color=0)
            if self.transform:
                dummy = self.transform(dummy)
            return dummy, label


def get_transforms():
    """Retourne les transforms train et val/test."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def get_loaders(csv_path: str, img_dir: str,
                batch_size: int = 64, seed: int = 42):
    """
    Scan, split 80/10/10 et création des DataLoaders.
    Retourne : train_loader, val_loader, test_loader, classes
    """
    # Reproducibilité
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Scan
    clean_df = scan_valid_images(csv_path, img_dir)
    classes = sorted(clean_df['Class'].unique().tolist())
    print(f"Classes : {len(classes)}")

    # Split 80 / 10 / 10 stratifié
    train_val_df, test_df = train_test_split(
        clean_df, test_size=0.10, random_state=seed, stratify=clean_df['Class']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1111, random_state=seed,
        stratify=train_val_df['Class']
    )
    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    train_tf, eval_tf = get_transforms()

    train_dataset = CoinDataset(train_df, img_dir, train_tf, classes)
    val_dataset   = CoinDataset(val_df,   img_dir, eval_tf,  classes)
    test_dataset  = CoinDataset(test_df,  img_dir, eval_tf,  classes)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader, classes
