import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# 1. Dummy Jahrom orchard data
# -----------------------------
class DummyJahromTreeDataset(Dataset):
    """
    Dummy dataset that mimics small satellite/drone patches of Jahrom orchards.

    Each sample:
        - img: 3 x H x W (float32, 0–1)
        - target: scalar tree count (float32)

    We simulate orchards by drawing small circular blobs on a blank background.
    """

    def __init__(self, length: int = 200, image_size: int = 128,
                 min_trees: int = 5, max_trees: int = 40):
        self.length = length
        self.image_size = image_size
        self.min_trees = min_trees
        self.max_trees = max_trees

    def __len__(self) -> int:
        return self.length

    def _draw_tree_blob(self, img: np.ndarray, x: int, y: int, radius: int):
        """Draw a simple circular 'tree' blob on a single-channel image."""
        h, w = img.shape
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        img[mask] = 1.0  # bright spot for tree canopy

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        H = W = self.image_size

        # Background (single channel)
        base = np.zeros((H, W), dtype=np.float32)

        # Random number of trees
        tree_count = random.randint(self.min_trees, self.max_trees)

        for _ in range(tree_count):
            # Random position within the patch
            x = random.randint(5, W - 5)
            y = random.randint(5, H - 5)

            # Small random radius (simulating canopy size)
            radius = random.randint(2, 4)
            self._draw_tree_blob(base, x, y, radius)

        # Add some Gaussian noise and normalize to [0,1]
        noise = np.random.normal(0, 0.05, size=base.shape).astype(np.float32)
        img = np.clip(base + noise, 0.0, 1.0)

        # Stack to 3 channels (RGB-like)
        img_3c = np.stack([img, img, img], axis=0)  # shape: (3, H, W)

        # Convert to tensors
        img_tensor = torch.from_numpy(img_3c)
        target_tensor = torch.tensor([float(tree_count)], dtype=torch.float32)

        return img_tensor, target_tensor


# -----------------------------
# 2. Simple CNN regressor
# -----------------------------
class TreeCountCNN(nn.Module):
    """
    Small CNN for regressing tree counts from image patches.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),  # scalar tree count
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x


# -----------------------------
# 3. Training loop
# -----------------------------
def train(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Using device: {device}")

    # Dataset & loaders
    train_ds = DummyJahromTreeDataset(length=300)
    val_ds = DummyJahromTreeDataset(length=60)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = TreeCountCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_train = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")

        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # MAE
            mae = torch.abs(preds.detach() - targets).mean().item()

            batch_size_curr = imgs.size(0)
            train_loss += loss.item() * batch_size_curr
            train_mae += mae * batch_size_curr
            n_train += batch_size_curr

            loop.set_postfix(loss=loss.item(), mae=mae)

        avg_train_loss = train_loss / n_train
        avg_train_mae = train_mae / n_train

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                preds = model(imgs)
                loss = criterion(preds, targets)
                mae = torch.abs(preds - targets).mean().item()

                batch_size_curr = imgs.size(0)
                val_loss += loss.item() * batch_size_curr
                val_mae += mae * batch_size_curr
                n_val += batch_size_curr

        avg_val_loss = val_loss / n_val
        avg_val_mae = val_mae / n_val

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"train_MAE={avg_train_mae:.2f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_MAE={avg_val_mae:.2f}"
        )


if __name__ == "__main__":
    train()
