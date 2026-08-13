"""The same training script written correctly. torch-guard reports nothing here.

This file is used as a regression test: any new rule that fires on it is producing a
false positive on idiomatic PyTorch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # No final activation: CrossEntropyLoss consumes raw logits.
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def train(model, dataset, val_dataset, device):
    loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    total_loss = 0.0

    for epoch in range(10):
        model.train()
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()

            # Scalars only: the graph is freed at the end of each iteration.
            losses.append(loss.item())
            total_loss += loss.item()

        validate(model, val_loader, criterion, device)

    return losses, total_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        total += criterion(logits, targets.to(device)).item()
    return total / max(len(loader), 1)
